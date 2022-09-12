function [ dlZ, state ] = forwardEncoder( self, encoder, dlX )
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
    dlZ = self.maskZ( dlZ );

end