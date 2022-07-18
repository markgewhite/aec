function Zq = interpRandSeries( x, xq, n, d, dcov )
    % Interpolate at query points from a generated random series
    arguments
        x       double  % points for random number generation
        xq      double  % sampling points following number generation
        n       double  % number of observation sets
        d       double  % number of dimensions for each set
        dcov    double  % covariance dimension (default 1)
    end

    switch dcov
        case 1
            Z = randSeries( n, d );
        case 2
            Z = randSeries( d, n )';
    end
    
    Zq = zeros( length(xq), d );
    
    for k = 1:d
        Zq( :, k ) = interp1( x, Z( :, k ), xq, 'cubic' );
    end

end