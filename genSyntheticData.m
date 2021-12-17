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
%           beta        : array of proportional factors between levels
%                         (must match the number basis elements)
%           constraints : array of times and values for smooth function
%           
% Outputs:
%           X     : synthetic time series
%           XFd   : functional synthetic data
%           Z     : coefficients
%
% ************************************************************************

function [ X, XFd, Z ] = genSyntheticData( nObs, nDim, basis, beta )

% prepare (optionally) for multiple basis layers
if isa( basis, 'basis')
    nLevels = 1;
    basisFd{1} = basis;
else
    nLevels = length( basis );
    basisFd = cell( nLevels, 1 );
    for i = 1:nLevels
        basisFd{i} = basis{i};
    end
end

% initialise the number of coefficients and spacing
nCoeff = zeros( nLevels, 1 );
tSpan = cell( nLevels, 1 );
for i = 1:nLevels
    nCoeff(i) = getnbasis( basisFd{i} );
    range = getbasisrange( basisFd{i} );
    tSpan{i} = linspace( range(1), range(2), nCoeff(i) )';
end
dt = tSpan{1}(2)-tSpan{1}(1);

Z = zeros( nCoeff(1), sum(nObs), nDim );

a = 0;
for c = 1:length(nObs)

    % generate random template function coefficients
    % with covariance between the dimensions (arguments reversed)
    coeff = randSeries( nCoeff(1), nDim );

    % add interpolated layers (if required)
    for i = 2:nLevels
        coeffi = randSeries( nCoeff(i), nDim );
        for j = 1:nDim
            coeff(:,j) = coeff(:,j) + ...
                           beta(i-1)*interp1( tSpan{i}, ...
                                              coeffi(:,j), tSpan{1}, ...
                                              'cubic' );
        end
    end
   
    for i = 1:nObs(c)
       
        % vary the template function 
        % with covariance between series elements 
        a = a+1;
        Z( :, a, : ) = coeff + beta(end)*randSeries( nDim , nCoeff(1) )';

        % warp the time domain at the top level, ensuring monotonicity
        monotonic = false;
        m = 0;
        while ~monotonic
            tWarp = tSpan{nLevels}+dt*randSeries( 1, length(tSpan{nLevels}) )';
            tWarp = interp1( tSpan{nLevels}, tWarp, tSpan{1}, 'cubic' );
            monotonic = all( diff(tWarp)>0 );
            m = m + 1;
        end
        if m > 1 
            disp( num2str(m) );
        end

        % interpolate the coefficients to the warped time points
        Z( :, a, : ) = interp1( tSpan{1}, Z(:,a,:), tWarp, 'cubic' );
               
    end

end

XFd = fd( Z, basisFd{1} );

X = eval_fd( tSpan{1}, XFd );

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

