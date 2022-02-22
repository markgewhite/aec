% ************************************************************************
% Function: fitKernels
%
% Generate random convolutional kernels (MINI-ROCKET)
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

function K = fitKernels( X, nFeatures )

inputLength = size( X, 2 );
K.nKernels = 84;
K.length = 9;
maxDilationsPerKernel = 32;

[ K.dilations, K.nFeaturesPerDilation ] = fitDilations( ...
                inputLength, K.nKernels, nFeatures, ...
                K.length, maxDilationsPerKernel );

nFeaturesPerKernel = sum( K.nFeaturesPerDilation );

Q = quantiles( K.nKernels*nFeaturesPerKernel );

K.biases = fitBiases( X, K.nKernels, ...
                         K.length, K.dilations, ...
                         K.nFeaturesPerDilation, Q );

end


function [ dil, nFeatPerD ] = fitDilations( inLen, nKernels, nFeat, kLen, maxDilPerK )

    nFeatPerK = fix( nFeat/nKernels );
    trueMaxDilPerK = min( nFeatPerK, maxDilPerK );
    multiplier = nFeatPerK / trueMaxDilPerK;

    maxExp = log2( (inLen - 1) / (kLen - 1));
    dil = fix(2.^(linspace(0,maxExp,trueMaxDilPerK)));
    nFeatPerD = groupcounts( dil' );
    nFeatPerD = fix( nFeatPerD*multiplier ); % this is a vector
    dil = unique( dil );

    remainder = nFeatPerK - sum( nFeatPerD );
    i = 1;
    while remainder > 0
        nFeatPerD(i) = nFeatPerD(i) + 1;
        remainder = remainder - 1;
        i = mod( i, length(nFeatPerD) ) + 1;
    end

end


function Q = quantiles( n )

    % low-discrepancy sequence to assign quantiles to kernel/dilation combinations
    Q = mod( (0:n+1)*(sqrt(5) + 1)/2, 1 );

end


function biases = fitBiases( X, nKernels, kLen, dilations, nFeatPerD, Q )

    [ nObs, inLen ] = size( X );

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
    biases = zeros( nFeat, 1 );

    featIdxStart = 1;

    for dilIdx = 1:nDil

        dilation = dilations( dilIdx );
        padding = fix( ((kLen - 1) * dilation)/2 );

        nFeatThisD = nFeatPerD( dilIdx );

        for kIdx = 1:nKernels

            featIdxEnd = featIdxStart + nFeatThisD - 1;

            x = X( randi(nObs), : );

            A = -x;        % A = alpha * X = -X
            G = x + x + x; % G = gamma * X = 3X

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

            cIdx = indices( kIdx, : );

            C = C_alpha + ...
                  C_gamma(cIdx(1),:)+C_gamma(cIdx(2),:)+C_gamma(cIdx(3),:);

            biases( featIdxStart:featIdxEnd ) = ...
                quantile( C, Q(featIdxStart:featIdxEnd) );

            featIdxStart = featIdxEnd + 1;

        end

    end

end



