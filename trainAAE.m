% ************************************************************************
% Function: trainAAE
%
% Train a bespoke adversarial autoencoder
%
% Parameters:
%           trnX        : training data
%           setup       : structure of all training/network parameters
%           
% Outputs:
%           dlnetEnc    : trained encoder network
%           dlnetDec    : trained decoder network
%
% ************************************************************************

function [ dlnetEnc, dlnetDec ] = trainAAE( trnX, setup, ax )

% create datastores
dsTrnX = arrayDatastore( trnX, 'IterationDimension', 2 );

% define the networks
[ dlnetEnc, dlnetDec ] = setup.designFcn( setup.enc, setup.dec );       

% setup the batches
mbqTrn = minibatchqueue(  dsTrnX,...
                          'MiniBatchSize', setup.batchSize, ...
                          'PartialMiniBatch', 'discard', ...
                          'MiniBatchFormat', 'CB' );

% initialise training parameters
avgG.enc = []; 
avgG.dec = []; 
avgGS.enc = [];
avgGS.dec = [];

nIter = floor( size(trnX,2)/setup.batchSize );
j = 0;
loss = zeros( nIter*setup.nEpochs, 3 );
fprintf('Training AAE (%d epochs): \n', setup.nEpochs );

for epoch = 1:setup.nEpochs
    
    % Shuffle the data
    shuffle( mbqTrn );

    % Loop over mini-batches.
    for i = 1:nIter
        
        j = j + 1;
        
        % Read mini-batch of data
        dlXTrn = next( mbqTrn );
        
        % Evaluate the model gradients 
        [ grad, state, loss(j,:), score ] = ...
                          dlfeval(  setup.gradFcn, ...
                                    dlnetEnc, ...
                                    dlnetDec, ...
                                    dlXTrn, ...
                                    setup );
        dlnetEnc.State = state.enc;
        dlnetDec.State = state.dec;

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
        
        % Update the generator network parameters
        [ dlnetEnc, avgG.enc, avgGS.enc ] = ...
                            adamupdate( dlnetEnc, ...
                                        grad.enc, ...
                                        avgG.enc, ...
                                        avgGS.enc, ...
                                        j, ...
                                        setup.enc.learnRate, ...
                                        setup.beta1, ...
                                        setup.beta2 );
        
        

    end

    % update progress on screen
    if mod( epoch, setup.valFreq )==0
        meanLoss = mean(loss( j-nIter+1:j, : ));
        fprintf('Loss (%d) = %1.3f  %1.3f  %1.3f\n', epoch, meanLoss );
        dlZTrn = predict( dlnetEnc, dlXTrn );
        ZTrn = double(extractdata( dlZTrn ));
        plotLatentComp( ax.ae.comp, dlnetDec, ZTrn, ...
                    setup.fda.tSpan, setup.fda.fdPar );
        drawnow;
    end

    % update learning rates
    if mod( epoch, setup.lrFreq )==0
        setup.enc.learnRate = setup.enc.learnRate*setup.lrFactor;
        setup.dec.learnRate = setup.dec.learnRate*setup.lrFactor;
    end

end


end

