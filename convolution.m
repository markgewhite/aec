% ************************************************************************
% Function: convolution
%
% Compute the convolution matrices, C_alpha and C_gamma
% Code adapted from original Python implementation.
%
% Parameters:
%           x        : vector of time series (input)
%           kLen     : kernel length
%           inLen    : input length
%           padding  : number of padding elements at each end
%           dilation : specified dilation
%           
% Outputs:
%           C_alpha  : convolution matrix
%           C_gamma  : convolution matrix
%
% ************************************************************************


function [C_alpha, C_gamma] = convolution( x, kLen, inLen, padding, dilation )

% obtain alpha and gamme vectors efficiently
A = -x';        % A = alpha * X = -X
G = x + x + x;  % G = gamma * X = 3X

C_alpha = A;

C_gamma = zeros( kLen, inLen );
C_gamma( ceil(kLen/2), : ) = G; % middle row

t1 = dilation;
t2 = inLen - padding;
 
% loop over the top half
for g = 1:fix(kLen/2)

    C_alpha( inLen-t2+1:end ) = C_alpha( inLen-t2+1:end ) + A( 1:t2 );
    C_gamma( g, inLen-t2+1:end ) = G( 1:t2 );

    t2 = t2 + dilation;

end

% loop over the bottom half
for g = ceil(kLen/2)+1:kLen

    C_alpha( 1:inLen-t1+1 ) = C_alpha( 1:inLen-t1+1 ) + A( t1:end );
    C_gamma( g, 1:inLen-t1+1 ) = G( t1:end );

    t1 = t1 + dilation;

end


end