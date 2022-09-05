function dlZ = maskZ( self, dlZ )
    % Mask latent codes based on the number of active dimensions
    arguments
        self            SubAEModel
        dlZ             {mustBeA(dlZ, {'double', 'dlarray'})}
    end

    for i = self.ZDimActive+1:self.ZDim
        dlZ(i,:) = 0;
    end

end