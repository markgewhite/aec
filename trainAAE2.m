% ************************************************************************
% Function: trainAAE2
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
%
% ************************************************************************

function [ dlnetEnc, dlnetDec ] = trainAAE2( trnX, trnC, setup, ax )


% define the networks
[ dlnetEnc, dlnetDec, dlnetDis ] = ...
        setup.designFcn( setup.enc, setup.dec, setup.dis );   

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
avgG.enc = []; 
avgG.dec = [];
avgG.dis = []; 
avgGS.enc = [];
avgGS.dec = [];
avgGS.dis = []; 

nIter = floor( size(trnX,2)/setup.batchSize );
j = 0;
loss = zeros( nIter*setup.nEpochs, 5 );
fprintf('Training AAE (%d epochs): \n', setup.nEpochs );

for epoch = 1:setup.nEpochs
    
    % Shuffle the data
    shuffle( mbqTrn );

    % Loop over mini-batches.
    for i = 1:nIter
        
        j = j + 1;
        
        % Read mini-batch of data
        [dlXTrn, dlCTrn] = next( mbqTrn );
        
        % Evaluate the model gradients 
        [ grad, state, loss(j,:), score ] = ...
                          dlfeval(  setup.gradFcn, ...
                                    dlnetEnc, ...
                                    dlnetDec, ...
                                    dlnetDis, ...
                                    dlXTrn, ...
                                    dlCTrn, ...
                                    setup );
        dlnetEnc.State = state.enc;
        dlnetDec.State = state.dec;
        dlnetDis.State = state.dis;

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
        
        % Update the generator network parameters
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

    % update progress on screen
    if mod( epoch, setup.valFreq )==0
        meanLoss = mean(loss( j-nIter+1:j, : ));
        fprintf('Loss (%d) = %1.3f  %1.3f  %1.3f %1.3f  %1.3f\n', epoch, meanLoss );
        dlZTrn = predict( dlnetEnc, dlXTrn );
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
    end

end


end

