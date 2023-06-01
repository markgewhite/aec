function [ outputs, state ] = forwardDecoder( self, decoder, dlZ )
    % Override to include outputs from branch outputs
    arguments
        self        BranchedModel
        decoder     dlnetwork
        dlZ         dlarray
    end

    % generate the list of component output layers
    layers = strings( self.ZDimAux+1, 1 );
    layers(1) = 'add';
    for i = 1:self.ZDimAux
        layers(i+1) = ['comp' num2str(i) '00'];
    end

    % reconstruct curves from latent codes
    [varargout{1:self.ZDimAux+2}] = forward( decoder, dlZ, Outputs = layers );

    % extract the variable number of outputs
    outputs.dlXGen = varargout{1};
    for i = 1:self.ZDimAux
        outputs.(['dlXB' num2str(i)]) = varargout{i+1};
    end
    state = varargout{end};

end