function Z = randSeries( n, d )
    % Generate a random series with covariance
    arguments
        n   double % number of points
        d   double % number of dimensions
    end

    % generate Gaussian random matrix
    R = randn( d );
    % get the covariance matrix with a fractional offset to avoid errors
    sigma = cov( R ) + 1E-6*eye( d );
    % get the means
    mu = mean( R );
    
    % generate coefficients from this random multivariate distribution
    Z = mvnrnd( mu, sigma, n );

end