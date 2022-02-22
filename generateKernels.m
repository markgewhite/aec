% ************************************************************************
% Function: generateKernels
%
% Generate random convolutional kernels (ROCKET)
% Code adapted from original Python implementation.
%
% Parameters:
%           nPoints : number of points in each time series
%           nKernels : how many kernels to generate
%           
% Outputs:
%           kernels : kernels data structure
%
% ************************************************************************

function kernels = generateKernels( nPoints, setup )

start = setup.candidateStart*2 + 1; % odd numbers

candidateLengths = linspace( start, start+(setup.nCandidates-1)*2, ...
                             setup.nCandidates );
kernels.lengths = candidateLengths(   ...
                    randi( length(candidateLengths), setup.nKernels, 1) );

kernels.weights = zeros( sum( kernels.lengths ), 1 );
kernels.biases = zeros( setup.nKernels, 1 );
kernels.dilations = zeros( setup.nKernels, 1 );
kernels.paddings = zeros( setup.nKernels, 1 );
kernels.correlations = zeros( setup.nKernels, 1 );

a1 = 1;

for i = 1:setup.nKernels

    l = kernels.lengths(i);

    if setup.isInterdependent
        w = randSeries( 1, l )';
    else
        w = randn( l, 1 );
    end

    if setup.smooth
        w = smooth( w );
    end

    b1 = a1 + l - 1;
    kernels.weights( a1:b1 ) = w - mean(w);
    kernels.correlations(i) = corr( w(1:end-1), w(2:end) );

    kernels.biases(i) = 2*rand - 1;

    kernels.dilations(i) = fix( 2^(rand*log2( (nPoints - 1) / (l - 1))) );

    if randi(2) == 1
        kernels.paddings(i) = fix( (l - 1) * kernels.dilations(i)/2 );
    else
        kernels.paddings(i) = 0;
    end

    a1 = b1 + 1;

end

end

function wHat = smooth( w )
 
    basisOrder = 4;
    penaltyOrder = 2;
    lambda = 1E-3; % 1E2
    nBasis = length(w)+penaltyOrder;
    tSpan = 1:length(w);
    basisFd = create_bspline_basis( ...
                            [ tSpan(1), tSpan(end) ], ...
                              nBasis, basisOrder);
    params = fdPar( basisFd, ...
                             penaltyOrder, ...
                             lambda );
    % smooth the data
    wFd = smooth_basis( tSpan, w, params );
    
    % re-sample it
    wHat = eval_fd( tSpan, wFd );

end
