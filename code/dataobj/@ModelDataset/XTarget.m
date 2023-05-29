function X = XTarget( self, tSpan )
    % Generate the output from XFd for a specified time span
    arguments
        self        ModelDataset
        tSpan       double
    end

    XCell = self.processX( tSpan, true, length(tSpan) );

    X = reshape( cell2mat( XCell ), [], self.NumObs, self.XChannels );

end