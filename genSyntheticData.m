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

function [ X, XFd, Z ] = genSyntheticData( nObs, nDim, ...
                                        basis, beta, constraints )

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

Z = zeros( nCoeff(1), sum(nObs), nDim );

a = 0;
for c = 1:length(nObs)

    % generate random template function coefficients
    % with covariance between the dimensions
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

    % adjust to fit constraints
    figure(1);
    ax1 = subplot(2,1,1);
    ax2 = subplot(2,1,2);
    X = eval_fd( tSpan{1}, fd( coeff, basisFd{1} ) );
    plot( ax1, tSpan{1}, X );
    coeff = fitToConstraints( coeff, constraints, basisFd{1} );
    X = eval_fd( tSpan{1}, fd( coeff, basisFd{1} ) );
    plot( ax2, tSpan{1}, X );
    
    for i = 1:nObs(c)
       
        % vary the template function 
        % with covariance between series elements
        a = a+1;
        Z( :, a, : ) = coeff + beta(end)*randSeries( nDim, nCoeff(1) )';
               
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


function Zhat = fitToConstraints( Z, C, BFd )

    [ nCoeff, nDim  ] = size( Z );
    dZ0 = zeros( nCoeff, 1 );
    dZ = Z;
    %options = optimset( 'Display', 'none', ...
    %                    'TolFun', 0.01, ...
    %                    'TolX', 0.001, ...
    %                    'PlotFcns', @optimplotfval) ;
    options = optimoptions( 'particleswarm', ...
                            'FunctionTolerance', 1E-6 );
    lb = -5*ones( nCoeff, 1 );
    ub = 5*ones( nCoeff, 1 );

    for j = 1:nDim
        objFcn = @(dZ) objective( dZ, Z( :, j ), C, BFd );
        %dZ( :, j ) = fminsearch( objFcn, dZ0, options );
        dZ( :, j ) = particleswarm( objFcn, nCoeff, lb, ub, options );
    end

    Zhat = Z + dZ;

end


function loss = objective( dZ, Z, C, BFd )

    % compute the loss from the offset dZ to Z for a given curve
    % according to the constraints C
    % with the curve defined from Z+dZ using the basis BFd

    % create the functional data object 
    XFd = fd( Z+dZ', BFd );

    % evalute the curve at the constraint points C
    XatC = eval_fd( C(:,1), XFd );

    % calculate the squared error of deviation from constraint points
    lossC = sum( (XatC - C(:,2)).^2 );

    % calculate the cost of the adjustment
    lossZ = sum( dZ.^2 );

    % compute the overall loss
    loss = lossC + 1E-9*lossZ;

end