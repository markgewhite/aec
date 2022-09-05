function self = train( self, thisData )
    % Train the autoencoder
    arguments
        self            SubAEModel
        thisData        ModelDataset
    end

    self = self.Trainer.runTraining( self, thisData );
    
    [self.AuxModelALE, self.ALEQuantiles, ...
        self.LatentComponents ] = self.getLatentResponse( thisData );

    if any(self.LossFcnTbl.Types=='Auxiliary')
        self.AuxNetworkALE = self.getAuxALE( thisData, auxFcn = @predictAuxNet );
    end

end