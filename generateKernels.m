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

function kernels = generateKernels( nPoints, nKernels, ...
                                    candidateStart, nCandidates )

if nargin < 3
    candidateStart = 7;
    nCandidates = 4;
else
    candidateStart = candidateStart*2 + 1;
end

candidateLengths = linspace( candidateStart, ...
                    candidateStart+(nCandidates-1)*2, nCandidates );
kernels.lengths = candidateLengths(   ...
                    randi( length(candidateLengths), nKernels, 1) );

kernels.weights = zeros( sum( kernels.lengths ), 1 );
kernels.biases = zeros( nKernels, 1 );
kernels.dilations = zeros( nKernels, 1 );
kernels.paddings = zeros( nKernels, 1 );

a1 = 1;

for i = 1:nKernels

    l = kernels.lengths(i);

    w = randn( l, 1 );

    b1 = a1 + l - 1;
    kernels.weights( a1:b1 ) = w - mean(w);

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

