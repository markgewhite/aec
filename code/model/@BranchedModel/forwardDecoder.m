function [ outputs, state ] = forwardDecoder( self, decoder, dlZ )
    % Override to include outputs from branch outputs
    arguments
        self                BranchedModel
        decoder             dlnetwork
        dlZ                 dlarray
    end

    % reconstruct curves from latent codes
    [varargout{1:self.ZDimAux+1}] = forward( decoder, dlZ );

    % extract the output
    % sum the components together
    outputs.dlXHat = varargout{1};
    outputs.dlXB = cell( self.ZDimAux, 1 );
    outputs.dlXB{1} = varargout{1};
    for i = 2:self.ZDimAux
        outputs.dlXB{i} = varargout{i};
        outputs.dlXHat = outputs.dlXHat + varargout{i};
    end
    state = varargout{end};

end