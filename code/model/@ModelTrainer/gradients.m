function [grad, state, loss] = gradients( nets, ...
                                          thisModel, ...
                                          dlXIn, ...
                                          dlXOut, ...
                                          dlP, ...
                                          dlY, ...
                                          preTraining )
    % Compute the model gradients
    % (Model object not supplied so nets can be traced)
    arguments
        nets         struct   % networks, separate for traceability
        thisModel    AEModel % contains all other relevant info
        dlXIn        dlarray  % input to the encoder
        dlXOut       dlarray  % output target for the decoder
        dlP          dlarray  % density estimation of X
        dlY          dlarray  % auxiliary outcome variable
        preTraining  logical  % flag indicating if in pretraining mode
    end

    % autoencoder training
    [ dlZGen, dlXGen, state ] = ...
                forward( thisModel, nets.Encoder, nets.Decoder, dlXIn );

    if thisModel.HasCentredDecoder
        % add the target mean to the prediction
        dlXGen = dlXGen + repmat( mean(dlXOut, 2), 1, size(dlXOut,2) );
    end

    % select the active loss functions
    isActive = thisModel.LossFcnTbl.DoCalcLoss;
    activeFcns = thisModel.LossFcnTbl( isActive, : );
    
    % compute the active loss functions in turn
    % and assign to networks
    
    nFcns = size( activeFcns, 1 );
    nLoss = sum( activeFcns.NumLosses );
    loss = zeros( nLoss, 1 );
    idx = 1;
    lossAccum = [];
    for i = 1:nFcns
       
        % identify the loss function
        thisName = activeFcns.Names(i);
        thisLossFcn = thisModel.LossFcns.(thisName);

        % assign indices for the number of losses returned
        lossIdx = idx:idx+thisLossFcn.NumLoss-1;
        idx = idx + thisLossFcn.NumLoss;

        % select the input variables
        switch thisLossFcn.Input
            case 'X-XHat'
                dlV = { dlXOut, dlXGen };
            case 'XHat'
                dlV = { dlXGen };
            case 'Z'
                dlV = { dlZGen };
            case 'P-Z'
                dlV = { dlP, dlZGen };
            case 'YHat'
                dlV = dlYHat;
            case 'Z-Y'
                dlV = { dlZGen, dlY };
            case 'X-Y'
                dlV = { dlXIn, dlY };
        end

        % calculate the loss
        % (make sure to use the model's copy 
        %  of the relevant network object)
        if thisLossFcn.HasNetwork
            % call the loss function with the network object
            thisNetwork = nets.(thisName);
            if thisLossFcn.HasState
                % and store the network state too
                [ thisLoss, state.(thisName) ] = ...
                        thisLossFcn.calcLoss( thisNetwork, dlV{:} );
            else
                thisLoss = thisLossFcn.calcLoss( thisNetwork, dlV{:} );
            end
        else
            % call the loss function straightforwardly
            thisLoss = thisLossFcn.calcLoss( dlV{:} );
        end
        loss( lossIdx ) = thisLoss;

        if thisLossFcn.UseLoss
            lossAccum = assignLosses( lossAccum, thisLossFcn, thisLoss, lossIdx );
        end

    end

    % compute the gradients for each network
    netNames = fieldnames( lossAccum );
    for i = 1:length(netNames)
    
        thisName = netNames{i};
        thisNetwork = nets.(thisName);
        grad.(thisName) = dlgradient( lossAccum.(thisName), ...
                                      thisNetwork.Learnables, ...
                                      'RetainData', true );
        
    end

end
