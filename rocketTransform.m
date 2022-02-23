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

[ inLen, nObs ] = size( X );
nKernels = kernels.kernels;
kLen = kernels.length;
nMetrics = kernels.metrics;
dilations = kernels.dilations;
nFeatPerD = kernels.featuresPerDilation;
biases = kernels.biases;

switch nMetrics
    case 1
        featureFcn = @compute1Feature;
    case 2
        featureFcn = @compute2Features;
    case 3
        featureFcn = @compute3Features;
    case 4
        featureFcn = @compute4Features;
    otherwise
        error('Number of metrics not within range.');
end

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

nFeat = nKernels*sum( nFeatPerD )*nMetrics;
features = zeros( nFeat, nObs );

for i = 1:nObs

    x = X( :, i );
    
    A = -x';        % A = alpha * X = -X
    G = x + x + x;  % G = gamma * X = 3X

    fIdxStart = 1;
    bIdx = 1;

    for dIdx = 1:nDil

        padding0 = mod( dIdx, 2 );
        dilation = dilations( dIdx );
        padding = fix( ((kLen - 1) * dilation)/2 );

        nFeatThisD = nFeatPerD( dIdx );

        C_alpha = A;

        C_gamma = zeros( kLen, inLen );
        C_gamma( ceil(kLen/2), : ) = G;

        Dstart = dilation;
        Dend = inLen - padding;

        for gIdx = 1:fix(kLen/2)

            C_alpha( inLen-Dend+1:end ) = ...
                C_alpha( inLen-Dend+1:end) + A( 1:Dend );
            C_gamma( gIdx, inLen-Dend+1:end ) = G( 1:Dend );

            Dend = Dend + dilation;

        end

        for gIdx = fix(kLen/2)+1:kLen

            C_alpha( 1:inLen-Dstart+1 ) = ...
                C_alpha( 1:inLen-Dstart+1 ) + A( Dstart:end );
            C_gamma( gIdx, 1:inLen-Dstart+1 ) = G( Dstart:end );

            Dstart = Dstart + dilation;

        end
    
        for kIdx = 1:nKernels
    
            fIdxEnd = fIdxStart + nFeatThisD*nMetrics - 1;
    
            padding1 = mod( padding0 + kIdx, 2 );

            cIdx = indices( kIdx, : );
    
            C = C_alpha + ...
                  C_gamma(cIdx(1),:)+C_gamma(cIdx(2),:)+C_gamma(cIdx(3),:);

            if padding1 == 0

                f1 = fIdxStart;
                f2 = f1+nMetrics-1;
                for j = 0:nFeatThisD-1

                    features( f1:f2, i) = ...
                        featureFcn( C, biases(bIdx + j), inLen);

                    f1 = f2+1;
                    f2 = f2+nMetrics;

                end

            else
                cLen = length(C)-2*padding;
                f1 = fIdxStart;
                f2 = f1+nMetrics-1;
                for j = 0:nFeatThisD-1

                    features( f1:f2, i ) = ...
                        featureFcn( C(padding+1:end-padding), ...
                                      biases(bIdx + j), cLen );
                    f1 = f2+1;
                    f2 = f2+nMetrics;

                end
            end
    
            fIdxStart = fIdxEnd + 1;
            bIdx = bIdx + nFeatThisD;
    
        end
    
    end

end

end


function z = compute1Feature( conv, bias, len )

    pos = conv > bias;
    ppv = sum( pos );

    z = ppv/len;

end


function z = compute2Features( conv, bias, len )

    pos = conv > bias;

    ppv = sum( pos );
    mpv = sum( conv(pos) );

    z = [ ppv mpv ]/len;

end


function z = compute3Features( conv, bias, len )

    pos = conv > bias;
    idx = 1:len;

    ppv = sum( pos );
    mpv = sum( conv(pos) );
    mipv = sum( idx(pos) );

    z = [ ppv mpv mipv ]/len;

end


function z = compute4Features( conv, bias, len )

    ppv = 0;
    mpv = 0;
    mipv = 0;
    cpv = 0;
    lspv = 0;

    for i = 1:len

        if conv(i) > bias
            ppv = ppv + 1;
            mpv = mpv + conv(i);
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

    z = [ ppv mpv mipv lspv ]/len;

end