function [grads, states, losses] = gradients( nets, ...
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
    [ vars, states ] = ...
                forward( thisModel, nets.Encoder, nets.Decoder, dlXIn );

    % add other variables
    vars.dlXOut = dlXOut;
    vars.dlY = dlY;

    if isfield( vars, 'dlXHat' )
        % determine the number of draws made if VAE
        nDraws = size( vars.dlXHat, 2 )/size( vars.dlXOut, 2 );
        if nDraws > 1
            % duplicate X & Y to match VAE's multiple draws
            vars.dlXOut = repmat( vars.dlXOut, 1, nDraws );
            vars.dlY = repmat( dlY, 1, nDraws );
        end
    end


    % select the active loss functions
    isActive = thisModel.LossFcnTbl.DoCalcLoss;
    isNonReconFcn = (thisModel.LossFcnTbl.Types~="Reconstruction");
    isActive( isNonReconFcn ) = isActive( isNonReconFcn ) ...
                            & repelem(~preTraining, sum(isNonReconFcn))';
    activeFcns = thisModel.LossFcnTbl( isActive, : );

    % compute the active loss functions in turn
    % and assign to networks
    
    nFcns = size( activeFcns, 1 );
    nLoss = sum( activeFcns.NumLosses );
    losses = zeros( nLoss, 1 );
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
        nInputs = length( thisLossFcn.Input );
        dlV = cell( nInputs, 1 );
        for j = 1:nInputs
            fld = thisLossFcn.Input{j};
            if isfield( vars, fld )
                dlV{j} = vars.(fld);
            else
                eid = 'LossFcn:NoForwardArg';
                msg = ['Model forward function does not return the required field: ' fld];
                throwAsCaller( MException(eid,msg) );
            end
        end

        % calculate the loss
        % (make sure to use the model's copy 
        %  of the relevant network object)
        if thisLossFcn.HasNetwork
            % call the loss function with the network object
            thisNetwork = nets.(thisName);
            if thisLossFcn.HasState
                % and store the network state too
                [ thisLoss, states.(thisName) ] = ...
                        thisLossFcn.calcLoss( thisNetwork, dlV{:} );
            else
                thisLoss = thisLossFcn.calcLoss( thisNetwork, dlV{:} );
            end
        else
            % call the loss function straightforwardly
            thisLoss = thisLossFcn.calcLoss( dlV{:} );
        end
        losses( lossIdx ) = thisLoss;

        if thisLossFcn.UseLoss
            lossAccum = assignLosses( lossAccum, thisLossFcn, thisLoss, lossIdx );
        end

    end

    % compute the gradients for each network
    netNames = fieldnames( lossAccum );
    for i = 1:length(netNames)
    
        thisName = netNames{i};
        thisNetwork = nets.(thisName);
        grads.(thisName) = dlgradient( lossAccum.(thisName), ...
                                      thisNetwork.Learnables, ...
                                      'RetainData', true );
        
    end

end
