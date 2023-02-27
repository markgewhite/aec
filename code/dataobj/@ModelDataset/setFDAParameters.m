function [fdParams, lambda] = setFDAParameters( self, tSpan, X )
    % Setup the FDA parameters object
    arguments
        self            ModelDataset
        tSpan           double
        X               double = []
    end

    if ~isempty(X)
        % use a special basis with one knot per point
        basis1Fd = create_bspline_basis( [tSpan(1) tSpan(end)], ...
                                         length(tSpan), ...
                                         self.FDA.BasisOrder );

        % define Generalised Cross-Validation function
        gcvFcn = @(L) gcv( L, X, tSpan, basis1Fd, self.FDA.PenaltyOrder );
    
        % find the loglambda where GCV is minimized
        warning( 'off', 'Wid2:reduce' );
        logLambda = fminsearch( gcvFcn, 0 );
        warning( 'on', 'Wid2:reduce' );
        lambda = 10^round( logLambda, 1 );
    else
        lambda = self.FDA.Lambda;
    end

    % create a basis for smoothing with fewer functions
    % (one knot per point is too expensive)
    %nBasis = fix(sqrt(length(tSpan))*5);
    nBasis = length(tSpan);
    nBasis = min(max( nBasis, 2 ), length(tSpan)) + self.FDA.PenaltyOrder;

    basisFd = create_bspline_basis( [tSpan(1) tSpan(end)], ...
                                    nBasis, ...
                                    self.FDA.BasisOrder );

    % setup the smoothing parameters
    fdParams = fdPar( basisFd, self.FDA.PenaltyOrder, lambda );

end