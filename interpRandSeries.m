% ************************************************************************
% Function: interpRandSeries
%
% Interpolate at query points from a generated random series
% 
% Parameters:
%       x    : points for random number generation
%       xq   : sampling points following number generation
%       n    : number of observation sets
%       d    : number of dimensions for each set
%       dcov : covariance dimension (default 1)
%           
% Outputs:
%       Zq   : generated data points
%
% ************************************************************************

function Zq = interpRandSeries( x, xq, n, d, dcov )

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