% ************************************************************************
% Function: applyKernels
%
% Apply random convolutional kernels (ROCKET)
% Code adapted from original Python implementation.
%
% Parameters:
%           X       : data
%           kernels : initialised kernels data structure
%           
% Outputs:
%           XT      : transformed data
%
% ************************************************************************

function XT = applyKernels( X, kernels, method )

if nargin < 3
    method = 'Standard';
end

switch method
    case 'Standard'
        nFeatures = 2;
        transformFcn = @applyKernel;
    case 'PPV'
        nFeatures = 1;
        transformFcn = @applyKernelPPV;
    case 'Multi'
        nFeatures = 4;
        transformFcn = @applyKernelMulti;
end

nObs = size( X, 2 );
nKernels = length( kernels.lengths );

% standardize
X = (X - mean(X, 'all'))./std(X, [], 'all' );

XT = zeros( nKernels*nFeatures, nObs ); % 2 features per kernel

for i = 1:nObs

    a1 = 1; % for weights
    a2 = 1; % for features

    for j = 1:nKernels

        b1 = a1 + kernels.lengths(j) - 1;
        b2 = a2 + nFeatures - 1;

        XT( a2:b2, i ) = transformFcn( X( :, i ), ...
                                      kernels.weights( a1:b1 ), ...
                                      kernels.lengths(j), ...
                                      kernels.biases(j), ...
                                      kernels.dilations(j), ...
                                      kernels.paddings(j) );

        a1 = b1 + 1;
        a2 = b2 + 1;

    end

end

end


function XT = applyKernel( X, weight, len, ...
                                      bias, dilation, padding )

    % original ROCKET algorithm with improved performance

    X = [ zeros( padding, 1 ); X; zeros( padding, 1 ) ];
    
    ppv = 0;
    high = -Inf;
       
    nCalc = length( X ) - (len-1)*dilation;
    idxRng = 0:dilation:(len-1)*dilation;
    for i = 1:nCalc

        idxRng = idxRng + 1;
        conv = bias + sum( weight.*X(idxRng) );

        if conv > high
            high = conv;
        end

        if conv > 0
            ppv = ppv + 1;
        end

    end

    XT = [ ppv/nCalc, high ];

end


function XT = applyKernelPPV( X, weight, len, ...
                                      bias, dilation, padding )

    % original ROCKET algorithm returning PPV alone

    X = [ zeros( padding, 1 ); X; zeros( padding, 1 ) ];
    
    ppv = 0;
       
    nCalc = length( X ) - (len-1)*dilation;
    idxRng = 0:dilation:(len-1)*dilation;
    for i = 1:nCalc

        idxRng = idxRng + 1;
        conv = bias + sum( weight.*X(idxRng) );

        if conv > 0
            ppv = ppv + 1;
        end

    end

    XT = ppv/nCalc;

end


function XT = applyKernelMulti( X, weight, len, ...
                                      bias, dilation, padding )

    % Multi ROCKET algorithm

    X = [ zeros( padding, 1 ); X; zeros( padding, 1 ) ];

    ppv = 0;
    mpv = 0;
    mipv = 0;
    cpv = 0;
    lspv = 0;
       
    nCalc = length( X ) - (len-1)*dilation;
    idxRng = 0:dilation:(len-1)*dilation;
    for i = 1:nCalc

        idxRng = idxRng + 1;
        conv = bias + sum( weight.*X(idxRng) );

        if conv > 0
            ppv = ppv + 1;
            mpv = mpv + conv;
            mipv = mipv + i;
            cpv = cpv + 1;
        else
            if cpv > lspv
                lspv = cpv;
            end
            cpv = 0;
        end

    end

    if lspv == 0
        lspv = cpv;
    end

    XT = [ ppv mpv mipv lspv ]/nCalc;

end


