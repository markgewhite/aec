% ************************************************************************
% Function: rocketTransform
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

function features = rocketTransform( X, kernels )

[ nObs, inLen ] = size( X );
nKernels = kernels.nKernels;
kLen = kernels.length;
dilations = kernels.dilations;
nFeatPerD = kernels.nFeaturesPerDilation;
biases = kernels.biases;

% equivalent to:
% >>> from itertools import combinations
% >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
indices = [
    0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8, ...
    0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4, ...
    0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7, ...
    0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8, ...
    1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4, ...
    1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7, ...
    1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8, ...
    2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6, ...
    2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8, ...
    2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7, ...
    3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8, ...
    4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8 ] + 1;
indices = reshape( indices, nKernels, 3 );

nDil = length(dilations);

nFeat = nKernels*sum( nFeatPerD );
features = zeros( nObs, nFeat );

for i = 1:nObs

    x = X( i, : );
    
    A = -x;        % A = alpha * X = -X
    G = x + x + x; % G = gamma * X = 3X

    featIdxStart = 1;

    for dilIdx = 1:nDil

        padding0 = mod( dilIdx, 2 );
        dilation = dilations( dilIdx );
        padding = fix( ((kLen - 1) * dilation)/2 );

        nFeatThisD = nFeatPerD( dilIdx );

        C_alpha = A;

        C_gamma = zeros( kLen, inLen );
        C_gamma( fix(kLen/2), : ) = G;

        Dstart = dilation;
        Dend = inLen - padding;

        for gammaIdx = 1:fix(kLen/2)

            C_alpha( inLen-Dend+1:end ) = ...
                C_alpha( inLen-Dend+1:end) + A( 1:Dend );
            C_gamma( gammaIdx, inLen-Dend+1:end ) = G( 1:Dend );

            Dend = Dend + dilation;

        end

        for gammaIdx = fix(kLen/2)+1:kLen

            C_alpha( 1:inLen-Dstart+1 ) = ...
                C_alpha( 1:inLen-Dstart+1 ) + A( Dstart:end );
            C_gamma( gammaIdx, 1:inLen-Dstart+1 ) = G( Dstart:end );

            Dstart = Dstart + dilation;

        end
    
        for kIdx = 1:nKernels
    
            featIdxEnd = featIdxStart + nFeatThisD - 1;
    
            padding1 = mod( padding0 + kIdx, 2 );

            cIdx = indices( kIdx, : );
    
            C = C_alpha + ...
                  C_gamma(cIdx(1),:)+C_gamma(cIdx(2),:)+C_gamma(cIdx(3),:);

            if padding1 == 0
                for fIdx = 0:nFeatThisD-1
                    features( i, featIdxStart + fIdx) = ...
                                PPV(C, biases(featIdxStart + fIdx));
                end
            else
                for fIdx = 0:nFeatThisD-1
                    features( i, featIdxStart + fIdx - 1) = ...
                                PPV( C(padding:end-padding+1), ...
                                           biases(featIdxStart + fIdx));
                end
            end
    
            featIdxStart = featIdxEnd + 1;
    
        end
    
    end

end

end


function z = PPV(a, b)
    z = mean( a > b );
end


