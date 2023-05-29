function XMean = XTargetMean( self, tSpan )
    % Calculate the mean target curve for a specified time span
    arguments
        self        ModelDataset
        tSpan       double
    end

    if self.XChannels == 1
        XMean = mean( self.XTarget( tSpan ), 2 );
    else
        XMean = mean( self.XTarget( tSpan ), 2 );
        XMean = permute( XMean, [1 3 2] );
    end

end