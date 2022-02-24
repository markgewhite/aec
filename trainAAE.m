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
                            trainAAE( XG, XT, Y, setup, ax )


% define the networks
try
    [ dlnetEnc, dlnetDec, dlnetDis, dlnetCls ] = ...
        setup.designFcn( setup.enc, setup.dec, setup.dis, setup.cls );   
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
XGTrn = XG( :, training(cvPart) );
XTTrn = XT( :, training(cvPart) );
YTrn = Y( training(cvPart) );

% create datastores
dsXTTrn = arrayDatastore( XTTrn, 'IterationDimension', 2 );
dsXGTrn = arrayDatastore( XGTrn, 'IterationDimension', 2 );
dsYTrn = arrayDatastore( YTrn, 'IterationDimension', 1 );   
dsTrn = combine( dsXTTrn, dsXGTrn, dsYTrn );

% setup the batches
mbqTrn = minibatchqueue( dsTrn,...
                      'MiniBatchSize', setup.batchSize, ...
                      'PartialMiniBatch', 'discard', ...
                      'MiniBatchFormat', 'CB' );

% create validation set - straight to dlarray
dlXTVal = dlarray( XT( :, test(cvPart)  ), 'CB' );
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

nIter = floor( size(XGTrn,2)/setup.batchSize );

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
        [ dlXTTrn, dlXGTrn, dlYTrn ] = next( mbqTrn );
        
        % Evaluate the model gradients 
        [ grad, state, lossTrn(j,:) ] = ...
                          dlfeval(  setup.gradFcn, ...
                                    dlnetEnc, ...
                                    dlnetDec, ...
                                    dlnetDis, ...
                                    dlnetCls, ...
                                    dlXTTrn, ...
                                    dlXGTrn, ...
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
        dlZVal = getEncoding( dlnetEnc, dlXTVal, setup );
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
        plotLatentComp( ax.ae.comp, dlnetDec, ZTrn, setup.cDim, ...
                        setup.fda.tSpan, setup.fda.fdPar );
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



