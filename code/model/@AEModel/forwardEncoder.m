function [ outputs, state ] = forwardEncoder( self, encoder, dlX )
    % Forward-run the encoder network
    % dlnetworks are provided for tracing purposes 
    % rather than using the object's network definitions
    arguments
        self        AEModel
        encoder     dlnetwork
        dlX         dlarray
    end

    if self.FlattenInput && size( dlX, 3 ) > 1
        dlX = flattenDLArray( dlX );
    end

    % generate latent encodings
    [ dlZ, state ] = forward( encoder, dlX );

    % mask Z based on number of active dimensions
    outputs.dlZ = self.maskZ( dlZ );

    % use the reparameterization trick if VAE
    if self.IsVAE
        [ outputs.dlZ, outputs.dlMu, outputs.dlLogVar ] = ...
                reparameterize( outputs.dlZ, self.NumEncodingDraws );
    else
        outputs.dlMu = [];
        outputs.dlLogVar = [];
    end
    
end