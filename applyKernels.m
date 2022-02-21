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

function [ XT, conv ] = applyKernels( X, kernels )

nObs = size( X, 2 );
nKernels = length( kernels.lengths );

XT = zeros( nKernels*2, nObs ); % 2 features per kernel
conv = cell( nKernels, nObs ); % store convolutions

for i = 1:nObs

    a1 = 1; % for weights
    a2 = 1; % for features

    for j = 1:nKernels

        b1 = a1 + kernels.lengths(j) - 1;
        b2 = a2 + 1;

        [ XT( a2:b2, i ), conv{ j, i } ] = applyKernel( X( :, i ), ...
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


function [ XT, conv ] = applyKernel( X, weight, len, ...
                                      bias, dilation, padding )

    X = [ zeros( padding, 1 ); X; zeros( padding, 1 ) ];
    
    ppv = 0;
    high = -Inf;
       
    nCalc = length( X ) - (len-1)*dilation;
    idxRng = 0:dilation:(len-1)*dilation;
    conv = zeros( nCalc, 1 );
    for i = 1:nCalc

        idxRng = idxRng + 1;
        conv(i) = bias + sum( weight.*X(idxRng) );

        if conv(i) > high
            high = conv(i);
        end

        if conv(i) > 0
            ppv = ppv + 1;
        end

    end

    XT = [ ppv/nCalc, high ];

end



