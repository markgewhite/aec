function self = smoothRawData( self, XCell )
    % Smooth the raw data with functional data analysis
    arguments
        self        ModelDataset
        XCell       cell
    end

    % find the series lengths (capped at padLen)
    self.XLen = min( cellfun( @length, XCell ), self.Padding.Length );

    % pad the series for smoothing
    X = padData( XCell, self.Padding.Length, self.Padding.Value, ...
                 Same = self.Padding.Same, ...
                 Location = self.Padding.Location, ...
                 Anchoring = self.Padding.Anchoring );

    % find the appropriate smoothing parameters for the raw data
    self.FDA.FdParamsOriginal = self.setFDAParameters( self.TSpan.Original, X );

    % create the smooth functions from the original data
    self.XFd = smooth_basis( self.TSpan.Original, ...
                             double(X), ...
                             self.FDA.FdParamsOriginal );

    % re-sample for the input
    XInput = eval_fd( self.TSpan.Input, self.XFd );

    % setup the smoothing parameters
    if isempty( self.FDA.Lambda )
        % find the best lambda using the data
        [self.FDA.FdParamsInput, self.FDA.Lambda] = ...
                    self.setFDAParameters( self.TSpan.Input, XInput );
    else
        % use the prescribed lambda
        self.FDA.FdParamsInput = ...
                    self.setFDAParameters( self.TSpan.Input );
    
    end

end