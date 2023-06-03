function self = train( self, thisData )
    % Train the autoencoder
    arguments
        self                    AEModel
        thisData                ModelDataset
    end

    % complete initialization
    [self, thisData] = self.finalizeInit( thisData );

    % perform training using the trainer
    self = self.Trainer.runTraining( self, thisData );
    
    % generate the functional components
    dlZ = self.encode( thisData, convert = false );
    self.LatentComponents = self.calcLatentComponents( dlZ = dlZ, ...
                                                       convert = true ); 

end