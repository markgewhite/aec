function [self, thisDataset] = finalizeInit( self, thisDataset )
    % Post-construction model initialization
    % For tasks that depend on subclass initiation 
    % but apply to all models
    arguments
        self           PCAModel
        thisDataset    ModelDataset
    end

    % set the target dimension
    [self, thisDataset] = finalizeInit@RepresentationModel( self, thisDataset );

    % set the PCA response function
    self.LatentResponseFcn = @(Z) self.reconstruct( Z );

end