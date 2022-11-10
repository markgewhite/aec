function self = train( self, thisData )
    % Train the autoencoder
    arguments
        self                    AEModel
        thisData                ModelDataset
    end

    % initialize the encoder and decoder networks
    self.Nets.Encoder = self.initEncoder;
    self.Nets.Decoder = self.initDecoder;

    % initialize the loss function networks, if required
    self = self.initLossFcnNetworks;

    % perform training using the trainer
    if self.UseParallelProcessing
        self = self.Trainer.runTraining( self, thisData );
    self = self.Trainer.runTraining( self, thisData );
    
    % set other elements
    [self.AuxModelALE, self.ALEQuantiles, ...
        self.LatentComponents ] = self.getLatentResponse( thisData );

    if any(self.LossFcnTbl.Types=='Auxiliary')
        self.AuxNetworkALE = self.getAuxALE( thisData, auxFcn = @predictAuxNet );
    end

    if self.ShowPlots
        % generate final plots
        plotLatentComp( self, type = 'Smoothed', shading = true );
    
        plotALE( self, type = 'Model' );
    
        if any(self.LossFcnTbl.Types == 'Auxiliary')
            plotALE( self, type = 'Network' );
        end
    
        drawnow;
    end

end