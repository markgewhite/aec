function scale = scalingFactor( data )
    % Calculate the channel-wise scaling factor
    arguments
        data            double
    end
    
    scale = squeeze(mean(var( data )))';

end