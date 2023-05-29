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
    
    % setup the smoothing parameters if not prescribed
    if isempty( self.FDA.Lambda )
        % find the best lambda using the data
        [self.FDA.FdParamsInput, self.FDA.Lambda] = ...
                    self.setFDAParameters( self.TSpan.Input, X );
    else
        self.FDA.FdParamsInput = ...
                    self.setFDAParameters( self.TSpan.Input );
    
    end

    % create the smooth functions
    self.XFd = smooth_basis( self.TSpan.Input, ...
                             double(X), ...
                             self.FDA.FdParamsInput );

end