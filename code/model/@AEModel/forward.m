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

    if ~isempty( self.ActiveCompLossFcn )
        % prepare the Z values

        switch self.ComponentType
            case 'PDP'
                dlZC = self.prepPDP( outputs.dlZ, ...
                         sampling = self.ActiveCompLossFcn.Sampling, ...
                         nSample = self.ActiveCompLossFcn.NumSamples, ...
                         maxObs = self.ActiveCompLossFcn.MaxObservations );
            case 'ALE'
                [ dlZC, A, w ] = self.prepALE( outputs.dlZ, ...
                         sampling = self.ActiveCompLossFcn.Sampling, ...
                         nSample = self.ActiveCompLossFcn.NumSamples, ...
                         maxObs = self.ActiveCompLossFcn.MaxObservations );
        end

        % append to the Z list
        dlZ2 = [outputs.dlZ dlZC];
    end

    % reconstruct curves from latent codes
    [ outputs2, states.Decoder ] = self.forwardDecoder( decoder, dlZ2 );

    % combine all output fields together
    outputs = mergeStructs( outputs, outputs2 );

    if ~isfield( outputs, 'dlXGen' )
        % dlXGen is absent (branched model method)
        % create it for convenience here
        outputs.dlXGen = outputs.dlXHat;
    end

    if ~isempty( self.ActiveCompLossFcn )
        % extract the XC portion
        if self.XChannels==1
            outputs.dlXCHat = outputs.dlXGen( :, nObs+1:end );
        else
            outputs.dlXCHat = outputs.dlXGen( :, :, nObs+1:end );
        end
        % finish constructing the components
        switch self.ComponentType
            case 'PDP'
                outputs.dlXC = self.calcPDP( outputs.dlXCHat );
            case 'ALE'
                outputs.dlXC = self.calcALE( outputs.dlXCHat, A, w );
        end

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