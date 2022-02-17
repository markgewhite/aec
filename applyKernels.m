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

function XT = applyKernels( X, kernels )

nObs = size( X, 2 );
nKernels = length( kernels.lengths );

XT = zeros( nKernels*2, nObs ); % 2 features per kernel

for i = 1:nObs

    a1 = 1; % for weights
    a2 = 1; % for features

    for j = 1:nKernels

        b1 = a1 + kernels.lengths(j) - 1;
        b2 = a2 + 1;

        XT( a2:b2, i ) = applyKernel( X( :, i ), ...
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

    inLen = length( X );
    outLen = (inLen + 2*padding) - (len - 1)*dilation;
    
    ppv = 0;
    high = -Inf;
    
    endPt = inLen + padding - (len - 1)*dilation;
    
    for i = -padding+1:endPt

        total = bias;
        index = i;

        for j = 1:len

            if index > 0 && index < inLen
                total = total + weight(j)*X(index);
            end
            index = index + dilation;
        
        end

        if total > high
            high = total;
        end

        if total > 0
            ppv = ppv + 1;
        end

    end

    XT = [ ppv/outLen, high ];

end


