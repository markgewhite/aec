function fdParams = setFDAParameters( tSpan, ...
                                      basisOrder, penaltyOrder, lambda )
    % Setup the FDA parameters object
    arguments
        tSpan           double
        basisOrder      double {mustBeInteger, mustBePositive}
        penaltyOrder    double {mustBeInteger, mustBePositive}
        lambda          double
    end

    % create a basis for smoothing with a knot at each point
    % with one function per knot
    nBasis = fix(length(tSpan)/10) + penaltyOrder;

    basisFd = create_bspline_basis( [tSpan(1) tSpan(end)], ...
                                    nBasis, ...
                                    basisOrder );

    % setup the smoothing parameters
    fdParams = fdPar( basisFd, penaltyOrder, lambda );

end