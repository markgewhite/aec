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

function [ dlnetEnc, dlnetDec, dlnetDis, dlnetCls ] = ...
                            trainAAE( trnX, trnC, setup, ax )


% define the networks
[ dlnetEnc, dlnetDec, dlnetDis, dlnetCls ] = ...
        setup.designFcn( setup.enc, setup.dec, setup.dis, setup.cls );   

% create datastores
dsTrnX = arrayDatastore( trnX, 'IterationDimension', 2 );
dsTrnC = arrayDatastore( trnC, 'IterationDimension', 1 );   
dsTrn = combine( dsTrnX, dsTrnC );

% setup the batches
mbqTrn = minibatchqueue(  dsTrn,...
                          'MiniBatchSize', setup.batchSize, ...
                          'PartialMiniBatch', 'discard', ...
                          'MiniBatchFormat', 'CB' );

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

nIter = floor( size(trnX,2)/setup.batchSize );
j = 0;
loss = zeros( nIter*setup.nEpochs, 9 );
if setup.verbose
    fprintf('Training AAE (%d epochs): \n', setup.nEpochs );
end

for epoch = 1:setup.nEpochs
    
    % Pre-training
    setup.pretraining = epoch<=setup.nEpochsPretraining;

    % Shuffle the data
    shuffle( mbqTrn );

    % Loop over mini-batches.
    for i = 1:nIter
        
        j = j + 1;
        
        % Read mini-batch of data
        [dlXTrn, dlCTrn] = next( mbqTrn );
        
        % Evaluate the model gradients 
        [ grad, state, loss(j,:) ] = ...
                          dlfeval(  setup.gradFcn, ...
                                    dlnetEnc, ...
                                    dlnetDec, ...
                                    dlnetDis, ...
                                    dlnetCls, ...
                                    dlXTrn, ...
                                    dlCTrn, ...
                                    setup );
        dlnetEnc.State = state.enc;
        dlnetDec.State = state.dec;

        % Update the network parameters
        switch setup.optimizer
            case 'ADAM'
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

                % Update the classifier network parameters
                if strcmp( setup.classifier, 'Network' ) ...
                        && ~setup.pretraining
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

    % update progress on screen
    if setup.verbose && mod( epoch, setup.valFreq )==0
        meanLoss = mean(loss( j-nIter+1:j, : ));
        fprintf('Loss (%4d) = %6.3f  %1.3f  %1.3f %1.3f  %1.3f  %1.3f  %1.3f %1.3f  %1.3f\n', epoch, meanLoss );
        dlZTrn = predict( dlnetEnc, dlXTrn );
        if setup.variational
            if setup.useVarMean
                dlZTrn = dlZTrn( 1:setup.zDim, : );
            else
                dlZTrn = reparameterize( dlZTrn );
            end
        end
        ZTrn = double(extractdata( dlZTrn ));
        plotLatentComp( ax.ae.comp, dlnetDec, ZTrn, setup.cDim, ...
                    setup.fda.tSpan, setup.fda.fdPar );
        plotZDist( ax.ae.distZTrn, ZTrn, 'AE: Z Train', true );
        drawnow;
    end

    % update learning rates
    if mod( epoch, setup.lrFreq )==0
        setup.enc.learnRate = setup.enc.learnRate*setup.lrFactor;
        setup.dec.learnRate = setup.dec.learnRate*setup.lrFactor;
        if setup.adversarial
            setup.dis.learnRate = setup.dis.learnRate*setup.lrFactor;
        end
        if ~setup.pretraining
            setup.cls.learnRate = setup.cls.learnRate*setup.lrFactor;
        end
    end

end


end

