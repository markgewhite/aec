% ************************************************************************
% Function: trainAAE
%
% Train a bespoke adversarial autoencoder with discriminator
%
% Parameters:
%           trnX        : training data
%           trnC        : training classes
%           setup       : structure of all training/network parameters
%           
% Outputs:
%           dlnetEnc    : trained encoder network
%           dlnetDec    : trained decoder network
%           dlnetDis    : trained discriminator network
%           dlnetCls    : trained classifier network
%
% ************************************************************************

function [ dlnetEnc, dlnetDec, dlnetDis, dlnetCls, lossTrn, constraint ] = ...
                            trainAAE( X, XN, Y, setup, ax )


% define the networks
try
    [ dlnetEnc, dlnetDec ] = setup.autoencoderFcn( setup.enc, setup.dec );
    dlnetDis = setup.discriminatorFcn( setup.dis );
    dlnetCls = setup.classifierFcn( setup.cls ); 
catch
    dlnetEnc = [];
    dlnetDec = [];
    dlnetDis = [];
    dlnetCls = [];
    lossTrn = NaN;
    constraint = 1;
    return
end

% re-partition training to create a validation set
cvPart = cvpartition( Y, 'Holdout', 0.25 );

% create training set
XNTrn = XN( :, training(cvPart), : );
XTrn = X( :, training(cvPart) );
YTrn = Y( training(cvPart) );

% create the datastore for the input X
if setup.enc.embedding
    % X is a numeric array
    dsXTrn = arrayDatastore( XTrn, 'IterationDimension', 2 );
else
    % X is a cell array containing sequences of variable length
    dsXTrn = arrayDatastore( XTrn, 'IterationDimension', 1, ...
                             'OutputType', 'same' );
end
% create the datastore for the time-normalised output X
dsXNTrn = arrayDatastore( XNTrn, 'IterationDimension', 2 );
if size( XNTrn, 3 ) > 1
    XNfmt = 'SSCB';
else
    XNfmt = 'CB';
end

% create the datastore for the labels/outcomes
dsYTrn = arrayDatastore( YTrn, 'IterationDimension', 1 );   

% combine them
dsTrn = combine( dsXTrn, dsXNTrn, dsYTrn );


% setup the minibatch queues
if setup.enc.embedding
    mbqTrn = minibatchqueue( dsTrn,...
                      'MiniBatchSize', setup.batchSize, ...
                      'PartialMiniBatch', 'discard', ...
                      'MiniBatchFormat', {'CB', XNfmt, 'CB'} );

else
    mbqTrn = minibatchqueue(  dsTrn,...
                      'MiniBatchSize', setup.batchSize, ...
                      'PartialMiniBatch', 'discard', ...
                      'MiniBatchFcn', @preprocTrnSeqBatch, ...
                      'MiniBatchFormat', {'CTB', XNfmt, 'CB'} );
end


% create validation set - straight to dlarray
XVal = X( :, test(cvPart) );

dlXVal = dlarray( XVal, 'CB' );
dlYVal = dlarray( Y( test(cvPart)  ), 'CB' );

% initialise training parameters
switch setup.optimizer
    case 'ADAM'
        avgG.enc = []; 
        avgG.dec = [];
        avgG.dis = [];
        avgG.cls = [];
        avgGS.enc = [];
        avgGS.dec = [];
        avgGS.dis = [];
        avgGS.cls = [];
    case 'SGDM'
        vel.enc = [];
        vel.dec = [];
        vel.dis = [];
        vel.cls = [];
end

nIter = floor( size(XNTrn,2)/setup.batchSize );

j = 0;
v = 0;
vp = setup.valPatience;

lossTrn = zeros( nIter*setup.nEpochs, 10 );
lossVal = zeros( ceil(nIter*setup.nEpochs/setup.valFreq), 1 );

if setup.verbose
    fprintf('Training AAE (%d epochs): \n', setup.nEpochs );
end

for epoch = 1:setup.nEpochs
    
    % Pre-training
    setup.preTraining = epoch<=setup.nEpochsPretraining;

    % Shuffle the data
    shuffle( mbqTrn );

    % Loop over mini-batches.
    for i = 1:nIter
        
        j = j + 1;
        
        % Read mini-batch of data
        [ dlXTTrn, dlXNTrn, dlYTrn ] = next( mbqTrn );
        if size( XNTrn, 3 ) > 1
            dlXNTrn = dlarray( squeeze(dlXNTrn), 'SCB' );
        end
        
        % Evaluate the model gradients 
        [ grad, state, lossTrn(j,:) ] = ...
                          dlfeval(  setup.gradFcn, ...
                                    dlnetEnc, ...
                                    dlnetDec, ...
                                    dlnetDis, ...
                                    dlnetCls, ...
                                    dlXTTrn, ...
                                    dlXNTrn, ...
                                    dlYTrn, ...
                                    setup );


        % Update the network parameters
        switch setup.optimizer
            case 'ADAM'
                if setup.postTraining || setup.preTraining 

                    try
                        dlnetEnc.State = state.enc;
                        dlnetDec.State = state.dec;
                    catch
                        lossTrn = NaN;
                        constraint = 2;
                        return
                    end
                    % Update the decoder network parameters
                    [ dlnetDec, avgG.dec, avgGS.dec ] = ...
                                adamupdate( dlnetDec, ...
                                            grad.dec, ...
                                            avgG.dec, ...
                                            avgGS.dec, ...
                                            j, ...
                                            setup.dec.learnRate, ...
                                            setup.beta1, ...
                                            setup.beta2 );
    
                    % Update the encoder network parameters
                    [ dlnetEnc, avgG.enc, avgGS.enc ] = ...
                                adamupdate( dlnetEnc, ...
                                            grad.enc, ...
                                            avgG.enc, ...
                                            avgGS.enc, ...
                                            j, ...
                                            setup.enc.learnRate, ...
                                            setup.beta1, ...
                                            setup.beta2 );
    
                    % Update the discriminator network parameters
                    if setup.adversarial
                        dlnetDis.State = state.dis;
                        [ dlnetDis, avgG.dis, avgGS.dis ] = ...
                                        adamupdate( dlnetDis, ...
                                                    grad.dis, ...
                                                    avgG.dis, ...
                                                    avgGS.dis, ...
                                                    j, ...
                                                    setup.dis.learnRate, ...
                                                    setup.beta1, ...
                                                    setup.beta2 );
                    end

                end

                % Update the classifier network parameters
                if strcmp( setup.classifier, 'Network' ) ...
                        && ~setup.preTraining
                    dlnetCls.State = state.cls;
                    [ dlnetCls, avgG.cls, avgGS.cls ] = ...
                                    adamupdate( dlnetCls, ...
                                                grad.cls, ...
                                                avgG.cls, ...
                                                avgGS.cls, ...
                                                j, ...
                                                setup.cls.learnRate, ...
                                                setup.beta1, ...
                                                setup.beta2 );
                end

            case 'SGDM'
                if setup.postTraining || setup.preTraining 

                    dlnetEnc.State = state.enc;
                    dlnetDec.State = state.dec;
                    % Update the decoder network parameters
                    [ dlnetDec, vel.dec ] = ...
                                sgdmupdate( dlnetDec, ...
                                            grad.dec, ...
                                            vel.dec, ...
                                            setup.dec.learnRate );
    
                    % Update the encoder network parameters
                    [ dlnetEnc, vel.enc ] = ...
                                sgdmupdate( dlnetEnc, ...
                                            grad.enc, ...
                                            vel.enc, ...
                                            setup.enc.learnRate );
    
                    % Update the discriminator network parameters
                    if setup.adversarial
                        dlnetDis.State = state.dis;
                        [ dlnetDis, vel.dis ] = ...
                                sgdmupdate( dlnetDis, ...
                                            grad.dis, ...
                                            vel.dis, ...
                                            setup.dis.learnRate );
                    end

                end
                
                % Update the classifier network parameters
                if ~setup.pretraining
                    dlnetCls.State = state.cls;
                    [ dlnetCls, vel.cls ] = ...
                            sgdmupdate( dlnetCls, ...
                                        grad.cls, ...
                                        vel.cls, ...
                                        setup.cls.learnRate );
                end

        end
        

        


    end

    if ~setup.preTraining && mod( epoch, setup.valFreq )==0
        
        % run a validation check
        v = v + 1;
        dlZVal = getEncoding( dlnetEnc, dlXVal, setup );
        switch setup.validationFcn
            case 'Network'
                dlYHatVal = predict( dlnetCls, dlZVal );
                dlYHatVal = double( ...
                    onehotdecode( dlYHatVal, single(setup.cLabels), 1 ))' - 1;
                lossVal(v) = sum( dlYHatVal~=dlYVal )/length(dlYVal);

            case 'Fisher'
                ZVal = double(extractdata( dlZVal ));
                YVal = double(extractdata( dlYVal ));
                model = fitcdiscr( ZVal', YVal );
                lossVal(v) = loss( model, ZVal', YVal );
        end

        if v > 2*vp-1
            if mean(lossVal(v-2*vp+1:v-vp)) < mean(lossVal(v-vp+1:v))
                % no longer improving - stop training
                break
            end
        end

    end

    % update progress on screen
    if setup.verbose && mod( epoch, setup.updateFreq )==0
        meanLoss = mean(lossTrn( j-nIter+1:j, : ));
        fprintf('Loss (%4d) = %6.3f  %1.3f  %1.3f %1.3f  %1.3f  %1.3f  %1.3f  %1.3f  %1.3f  %1.3f', ...
            epoch, meanLoss );
        if setup.preTraining
            fprintf('\n');
        else
            fprintf(' : %1.3f\n', lossVal(v));
        end

        dlZTrn = getEncoding( dlnetEnc, dlXTTrn, setup );
        ZTrn = double(extractdata( dlZTrn ));
        for c = 1:setup.nChannels
            plotLatentComp( ax.ae.comp(:,c), dlnetDec, ZTrn, c, ...
                            setup.fda.tSpan, setup.fda.fdPar );
        end
        plotZDist( ax.ae.distZTrn, ZTrn, 'AE: Z Train', true );
        drawnow;
    end

    % update learning rates
    if mod( epoch, setup.lrFreq )==0
        if setup.postTraining || setup.preTraining 
            setup.enc.learnRate = setup.enc.learnRate*setup.lrFactor;
            setup.dec.learnRate = setup.dec.learnRate*setup.lrFactor;
            if setup.adversarial
                setup.dis.learnRate = setup.dis.learnRate*setup.lrFactor;
            end
        end
        if ~setup.preTraining
            setup.cls.learnRate = setup.cls.learnRate*setup.lrFactor;
        end
    end

end

constraint = -1;

end



