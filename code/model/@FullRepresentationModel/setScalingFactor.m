function self = setScalingFactor( self, data )
    % Set the scaling factors for reconstructions
    arguments
        self            FullRepresentationModel
        data            double
    end
    
    % set the channel-wise scaling factor
    self.Scale = squeeze(mean(var( data )))';

end