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

        % find L that minimizws gcvFcn to a precision of 0.1
        %opt = optimset( 'MaxIter', 12, 'Display', 'off' );
        %logLambda = fminsearch( gcvFcn, 0, opt );
        logLambda = fminsearch( gcvFcn, 0 );

        warning( 'on', 'Wid2:reduce' );
        lambda = 10^round( logLambda, 1 );
    else
        lambda = self.FDA.Lambda;
    end

    % create a basis for smoothing with fewer functions
    % (one knot per point is too costly in memory and computation)
    nBasis = fix( length(tSpan)/self.FDA.PtsPerKnot );
    nBasis = min(max( nBasis, 2 ), length(tSpan)) + self.FDA.PenaltyOrder;

    basisFd = create_bspline_basis( [tSpan(1) tSpan(end)], ...
                                    nBasis, ...
                                    self.FDA.BasisOrder );

    % setup the smoothing parameters
    fdParams = fdPar( basisFd, self.FDA.PenaltyOrder, lambda );

end