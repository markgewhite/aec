function [ dlZ, dlXGen, dlXHat, dlXC, state ] = forward( self, encoder, decoder, dlX )
    % Forward-run the encoder and decoder networks
    arguments
        self        AEModel
        encoder     dlnetwork
        decoder     dlnetwork
        dlX         dlarray
    end

    % generate latent encodings
    [ dlZ, state.Encoder ] = self.forwardEncoder( encoder, dlX );

    nObs = size( dlZ, 2 );

    idx = find( self.LossFcnTbl.Types=='Component'  ...
                & self.LossFcnTbl.DoCalcLoss );
    if ~isempty( idx )
        % active component loss functions are present
        % use the first one to determine parameters
        thisName = self.LossFcnTbl.Names(idx(1));
        thisLossFcn = self.LossFcns.(thisName);
        % prepare the Z values
        [~, ~, dlZC] = self.calcResponse( ...
                                dlZ, ...
                                mode = 'InputOnly', ...
                                sampling = thisLossFcn.Sampling, ...
                                nSample = thisLossFcn.NumSamples, ...
                                maxObs = thisLossFcn.MaxObservations);

        % append to the Z list
        dlZ2 = [dlZ dlZC];

    else
        dlZ2 = dlZ;

    end

    % reconstruct curves from latent codes
    [ dlXGen, state.Decoder ] = self.forwardDecoder( decoder, dlZ2 );

    if ~isempty( idx )
        % extract the XC portion
        dlXCHat = dlXGen( :, nObs+1:end, : );
        % finish constructing the components
        dlXC = self.calcResponse( ...
                                [], ...
                                mode = 'OutputOnly', ...
                                sampling = thisLossFcn.Sampling, ...
                                dlXC = dlXCHat );
    else
        dlXC = [];
    end

    dlXHat = dlXGen( :, 1:nObs, : );
    if self.HasCentredDecoder
        % add the target mean to the prediction
        if size( dlXGen, 3 )==1
            dlXHat = dlXHat + repmat( self.MeanCurveTarget, 1, size(dlXHat,2) );
        else
            dlXHat = dlXHat + repmat( self.MeanCurveTarget, 1, 1, size(dlXHat,3) );
        end
    end

end