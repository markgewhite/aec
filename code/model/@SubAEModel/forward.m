function [ dlZ, dlXHat, state ] = forward( self, encoder, decoder, dlX )
    % Forward-run the encoder and decoder networks
    arguments
        self        SubAEModel
        encoder     dlnetwork
        decoder     dlnetwork
        dlX         dlarray
    end

    % generate latent encodings
    [ dlZ, state.Encoder ] = self.forwardEncoder( encoder, dlX );

    % reconstruct curves from latent codes
    [ dlXHat, state.Decoder ] = self.forwardDecoder( decoder, dlZ );

end