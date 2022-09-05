function [XFd, XLen] = smoothRawData( XCell, padding, tSpan, fda )

    % find the series lengths (capped at padLen)
    XLen = min( cellfun( @length, XCell ), padding.Length );

    % pad the series for smoothing
    X = padData( XCell, padding.Length, padding.Value, ...
                 Same = padding.Same, ...
                 Location = padding.Location, ...
                 Anchoring = padding.Anchoring );
    
    % setup the smoothing parameters
    fdParams = setFDAParameters( tSpan, ...
                                 fda.BasisOrder, fda.PenaltyOrder, ...
                                 fda.Lambda );

    % create the smooth functions
    XFd = smooth_basis( tSpan, X, fdParams );

end