function [ dlXHat, state ] = forwardDecoder( self, decoder, dlZ )
    % Forward-run the decoder network
    % dlnetworks are provided for tracing purposes 
    % rather than using the object's network definitions
    arguments
        self        SubAEModel
        decoder     dlnetwork
        dlZ         dlarray
    end

    % reconstruct curves from latent codes
    [ dlXHat, state ] = forward( decoder, dlZ );

end