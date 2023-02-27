function err = gcv( logLambda, X, tSpan, basis, penaltyOrder  )
    % Objective function returning GCV error for given smoothing
    arguments
        logLambda       double
        X               double
        tSpan           double
        basis
        penaltyOrder    double
    end

    % set smoothing parameters
    XFdParam = fdPar( basis, penaltyOrder, 10^logLambda );
    
    % perform smoothing
    [~, ~, err] = smooth_basis( tSpan, X, XFdParam );

    err = mean(err, 'all');

end