% ************************************************************************
% Function: randSeries
%
% Generate a random series with covariance.
% The purpose is to give the points interdependence.
% 
% Parameters:
%           n : number of points
%           d : number of dimensions
%           
% Outputs:
%           Z : generated data points
%
% ************************************************************************

function Z = randSeries( n, d )

    % generate Gaussian random matrix
    R = randn( d );
    % get the covariance matrix with a fractional offset to avoid errors
    sigma = cov( R ) + 1E-6*eye( d );
    % get the means
    mu = mean( R );
    
    % generate coefficients from this random multivariate distribution
    Z = mvnrnd( mu, sigma, n );

end