function self = finalizeInit( self, thisDataset )
    % Post-construction model initialization
    % For tasks that depend on subclass initiation 
    % but apply to all models
    arguments
        self           AEModel
        thisDataset    ModelDataset
    end

    % set the target dimension
    self = finalizeInit@RepresentationModel( self, thisDataset );

    % initialize the encoder and decoder networks
    self.Nets.Encoder = self.initEncoder;
    self.Nets.Decoder = self.initDecoder;
    
    % initialize any other networks used by loss functions
    self = self.initLossFcnNetworks;

end
