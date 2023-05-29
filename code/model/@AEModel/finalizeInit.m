function [self, thisDataset] = finalizeInit( self, thisDataset )
    % Post-construction model initialization
    % For tasks that depend on subclass initiation 
    % but apply to all models
    arguments
        self           AEModel
        thisDataset    ModelDataset
    end

    % set the target dimension
    [self, thisDataset] = finalizeInit@RepresentationModel( self, thisDataset );

    % initialize the encoder and decoder networks
    self.Nets.Encoder = self.initEncoder;
    self.Nets.Decoder = self.initDecoder;

    % initialize the loss function networks, if required
    self = self.initLossFcnNetworks;

end