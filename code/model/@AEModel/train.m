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
    self = self.Trainer.runTraining( self, thisData );
    
    % get the auxiliary model's response to each Z element
    self = self.getAuxResponse( thisData );
    
    % generate the functional components
    self.LatentComponents = self.getLatentResponse( thisData );

end