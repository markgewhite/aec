function [XFd, XLen, lambda] = smoothRawData( self, XCell )
    % Smooth the raw data with functional data analysis
    arguments
        self        ModelDataset
        XCell       cell
    end

    % find the series lengths (capped at padLen)
    XLen = min( cellfun( @length, XCell ), self.Padding.Length );

    % pad the series for smoothing
    X = padData( XCell, self.Padding.Length, self.Padding.Value, ...
                 Same = self.Padding.Same, ...
                 Location = self.Padding.Location, ...
                 Anchoring = self.Padding.Anchoring );
    
    % setup the smoothing parameters
    if isempty( self.FDA.Lambda )
        % find the best lambda using the data
        [fdParams, lambda] = self.setFDAParameters( self.TSpan.Original, X );
    else
        % use the prescribed lambda
        fdParams = self.setFDAParameters( self.TSpan.Original );
        lambda = self.FDA.Lambda;
    end

    % create the smooth functions
    XFd = smooth_basis( self.TSpan.Original, double(X), fdParams );

end