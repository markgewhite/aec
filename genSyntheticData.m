% ************************************************************************
% Function: genFunctionalData
%
% Generate synthetic functional data 
% based originally on the method proposed by Hsieh et al. (2021).
%
% Enhanced with option to have multiple basis levels.
% The number of levels is specified if basis is a cell array
% 
% Parameters:
%           nObs        : number of observations per class (vector)
%           nDim        : number of dimensions
%           basis       : functional basis 
%                         (data object or cell array of data objects)
%           mu          : array of template magnitudes
%           sigma       : array of template magnitude variances
%           tau         : warping variance
%           
% Outputs:
%           X     : synthetic time series
%           XFd   : functional synthetic data
%           Z     : coefficients
%
% ************************************************************************

function [ X, XFd, Z ] = genSyntheticData( nObs, nDim, basis, ...
                                                mu, sigma, tau )

% prepare (optionally) for multiple basis layers
if isa( basis, 'basis')
    nLevels = 1;
    basisFd{1} = basis;
else
    nLevels = length( basis );
    basisFd = cell( nLevels, 1 );
    for j = 1:nLevels
        basisFd{j} = basis{j};
    end
end

% initialise the number of coefficients and spacing
% allow extra space either end for extrapolation when time warping
% the time domains are twice as long
nCoeff = zeros( nLevels, 1 );
tSpan = cell( nLevels, 1 );
for j = 1:nLevels
    nCoeff(j) = getnbasis( basisFd{j} );
    range = getbasisrange( basisFd{j} );
    extra = 0.5*(range(2)-range(1));
    tSpan{j} = linspace( range(1)-extra, range(2)+extra, 2*nCoeff(j) )';
end

% set find time domain which has the required length
tSpanFinal = linspace( range(1), range(2), nCoeff(1) )';

% calculate the time step for when calculating the warp gradient
dt = tSpan{1}(2)-tSpan{1}(1);

% initialise the template array across levels
template = zeros( 2*nCoeff(1), nDim, nLevels );

% initialise the array holding the generated data
Z = zeros( nCoeff(1), sum(nObs), nDim );

a = 0;
for c = 1:length(nObs)

    % generate random template function coefficients
    % with covariance between the series elements
    % interpolating to the base layer (1)
    for j = 1:nLevels
        template( :,:,j ) = interpRandSeries( tSpan{j}, tSpan{1}, ...
                                              2*nCoeff(j), nDim, 2 );
    end
   
    for i = 1:nObs(c)

        a = a+1;

        % vary the template function across levels
        sample = zeros( 2*nCoeff(1), nDim );
        for j = 1:nLevels 
            sample = sample + (mu(j) + sigma(j)*randn(1,1))*template( :,:,j );
        end

        % warp the time domain at the top level, ensuring monotonicity
        % and avoiding excessive curvature by constraining the gradient
        monotonic = false;
        excessCurvature = false;
        while ~monotonic || excessCurvature
            % generate a time warp series based on the top-level 
            tWarp = tSpan{end}+tau*randSeries( 1, length(tSpan{end}) )';
            % interpolate so it fits the initial level
            tWarp = interp1( tSpan{end}, tWarp, tSpan{1}, 'spline' );
            % check constraints
            grad = diff( tWarp )/dt;
            monotonic = all( grad>0 );
            excessCurvature = any( grad>2 ) | any( grad<0.2 );
        end

        % interpolate the coefficients to the warped time points
        Z( :, a, : ) = interp1( tWarp, sample, tSpanFinal, 'spline' );
               
    end

end

XFd = fd( Z, basisFd{1} );

X = eval_fd( tSpanFinal, XFd );

end


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


function Zq = interpRandSeries( x, xq, n, d, dcov )

    % x    : points for random number generation
    % xq   : sampling points following number generation
    % n    : number of observation sets
    % d    : number of dimensions for each set
    % dcov : covariance dimension (default 1)
    
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