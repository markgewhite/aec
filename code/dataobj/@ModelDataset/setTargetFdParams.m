function [ fdParams, lambda ] = setTargetFdParams( self, X )
    % Get the target FD parameters for a given data set
    arguments
        self        ModelDataset
        X           double
    end

    if length( self.TSpan.Target ) ~= size( X, 1 )
        error('X does not match the target timespan.');
    end

    tSpan = self.TSpan.Target;
    basis = create_bspline_basis( [tSpan(1) tSpan(end)], ...
                                  length(tSpan), ...
                                  self.FDA.BasisOrder);

    gcvFcn = @(L) gcv( L, X, tSpan, basis, self.FDA.PenaltyOrder );

    % find the loglambda where GCV is minimized
    logLambda = fminbnd( gcvFcn, -10, 10 );
    lambda = 10^round( logLambda, 1 );

    fdParams = setFDAParameters( tSpan, ...
                                 self.FDA.BasisOrder, ...
                                 self.FDA.PenaltyOrder, ...
                                 lambda );

end