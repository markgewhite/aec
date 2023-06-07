function [ outputs, states ] = forward( self, encoder, decoder, dlX )
    % Forward-run the encoder and decoder networks
    arguments
        self        BranchedModel
        encoder     dlnetwork
        decoder     dlnetwork
        dlX         dlarray
    end

    if ~strcmp( self.ComponentType, 'AEC' )
        % revert to non-branched method
        [ outputs, states ] = forward@AEModel( self, encoder, decoder, dlX );
        return
    end

    % generate latent encodings
    [ outputs, states.Encoder ] = self.forwardEncoder( encoder, dlX );

    outputs.dlZAux = outputs.dlZ( 1:self.ZDimAux, : );

    % reconstruct curves from latent codes
    [ outputs2, states.Decoder ] = self.forwardDecoder( decoder, outputs.dlZ );

    % combine all output fields together
    outputs = mergeStructs( outputs, outputs2 );

    if ~isempty( self.ActiveCompLossFcn )
        outputs.dlXC = self.calcAEC( outputs.dlXB, ...
                                     sampling = self.ActiveCompLossFcn.Sampling, ...
                                     nSample = self.ActiveCompLossFcn.NumSamples );
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