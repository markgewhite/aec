function [ outputs, states ] = forward( self, encoder, decoder, dlX )
    % Forward-run the encoder and decoder networks
    arguments
        self        AEModel
        encoder     dlnetwork
        decoder     dlnetwork
        dlX         dlarray
    end

    % generate latent encodings
    [ outputs, states.Encoder ] = self.forwardEncoder( encoder, dlX );

    outputs.dlZAux = outputs.dlZ( 1:self.ZDimAux, : );

    nObs = size( outputs.dlZ, 2 );

    idx = find( self.LossFcnTbl.Types=='Component'  ...
                & self.LossFcnTbl.DoCalcLoss );
    if ~isempty( idx )
        % active component loss functions are present
        % use the first one to determine parameters
        thisName = self.LossFcnTbl.Names(idx(1));
        thisLossFcn = self.LossFcns.(thisName);
        % prepare the Z values
        [~, ~, dlZC] = self.calcLatentComponents( ...
                                dlZ = outputs.dlZ, ...
                                mode = 'InputOnly', ...
                                sampling = thisLossFcn.Sampling, ...
                                nSample = thisLossFcn.NumSamples, ...
                                maxObs = thisLossFcn.MaxObservations);

        % append to the Z list
        dlZ2 = [outputs.dlZ dlZC];

    else
        dlZ2 = outputs.dlZ;

    end

    % reconstruct curves from latent codes
    [ outputs2, states.Decoder ] = self.forwardDecoder( decoder, dlZ2 );

    % combine all output fields together
    outputs = mergeStructs( outputs, outputs2 );

    if ~isempty( idx )
        % extract the XC portion
        if self.XChannels==1
            outputs.dlXCHat = outputs.dlXGen( :, nObs+1:end );
        else
            outputs.dlXCHat = outputs.dlXGen( :, :, nObs+1:end );
        end
        % finish constructing the components
        outputs.dlXC = self.calcLatentComponents( ...
                                mode = 'OutputOnly', ...
                                sampling = thisLossFcn.Sampling, ...
                                dlXC = outputs.dlXCHat );
    else
        outputs.dlXC = [];
    end

    if self.XChannels==1
        outputs.dlXHat = outputs.dlXGen( :, 1:nObs );
    else
        outputs.dlXHat = outputs.dlXGen( :, :, 1:nObs );
    end
    
    if self.HasCentredDecoder
        % add the target mean to the prediction
        if self.XChannels==1
            outputs.dlXHat = outputs.dlXHat ...
                + repmat( self.MeanCurveTarget, 1, size(outputs.dlXHat,2) );
        else
            outputs.dlXHat = outputs.dlXHat ...
                + repmat( self.MeanCurveTarget, 1, 1, size(outputs.dlXHat,3) );
        end
    end

end