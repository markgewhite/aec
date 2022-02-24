% ************************************************************************
% Function: rocketTransform
%
% Apply random convolutional kernels (ROCKET)
% Code adapted from original Python implementation.
%
% Parameters:
%           X          : data
%           parameters : initialised parameters
%           
% Outputs:
%           features   : transformed data
%           nDilTrunc  : number of dilations truncated
%
% ************************************************************************

function [ features, nDilTrunc ] = rocketTransform( X, parameters )

% extract parameters
nKernels = parameters.kernels;
kLen = parameters.length;
nMetrics = parameters.metrics;
dilations = parameters.dilations;
nFeatPerD = parameters.featuresPerDilation;
biases = parameters.biases;

% determine X size
lengthsVary = iscell( X );
if lengthsVary
    nObs = length( X );
else
    [ inLen, nObs ] = size( X );
end

% set the feature calculation function
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

% define the all possible kernel combinations 
% triplet combinations in the range 0-8.
% this matrix is taken from Python so one is added for Matlab indexing
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
nDilTrunc = 0;

nFeat = nKernels*sum( nFeatPerD )*nMetrics;
features = zeros( nFeat, nObs );

for i = 1:nObs

    % extract the next curve
    if lengthsVary
        x = X{ i };
        inLen = length( x ); %costly?
    else
        x = X( :, i );
    end

    % set dilation upper limit in case this time series is too short
    maxDil = fix( 2^log2( (inLen - 1) / (kLen - 1)) );
    
    % obtain alpha and gamme vectors efficiently
    A = -x';        % A = alpha * X = -X
    G = x + x + x;  % G = gamma * X = 3X

    f1 = 1; % feature index counter: 1st in block of metrics
    b = 1; % bias index counter

    % loop over all specified dilations for this curve
    for d = 1:nDil 

        % include padding on alternate dilations
        padding0 = mod( d, 2 );

        % extract the specified dilation
        dilation = dilations( d );
        if dilation > maxDil
            % time series to too short for reference dilation
            % find the largest dilation that is valid
            dilation = dilations( find(dilations<maxDil, 1, 'last') );
            if isempty(dilation)
                dilation = 1;
            end
            nDilTrunc = nDilTrunc + 1;
        end

        % set the padding in proportion
        padding = fix( ((kLen - 1) * dilation)/2 );

        % extract the number of features per dilation
        nFeatThisD = nFeatPerD( d ); 

        % calculate the convolution matrices
        C_alpha = A;

        C_gamma = zeros( kLen, inLen );
        C_gamma( ceil(kLen/2), : ) = G; % middle row

        t1 = dilation;
        t2 = inLen - padding;

        % loop over the top half
        for g = 1:fix(kLen/2)

            C_alpha( inLen-t2+1:end ) = ...
                C_alpha( inLen-t2+1:end) + A( 1:t2 );
            C_gamma( g, inLen-t2+1:end ) = G( 1:t2 );

            t2 = t2 + dilation;

        end

        % loop over the bottom half
        for g = fix(kLen/2)+1:kLen

            C_alpha( 1:inLen-t1+1 ) = ...
                C_alpha( 1:inLen-t1+1 ) + A( t1:end );
            C_gamma( g, 1:inLen-t1+1 ) = G( t1:end );

            t1 = t1 + dilation;

        end
    
        % compute the convolutions for each kernel
        for k = 1:nKernels
    
            % set the feature block end point
            f2 = f1 + nFeatThisD*nMetrics - 1;
    
            % alternate padding 
            padding1 = mod( padding0 + k, 2 );

            % combine alpha & gamma matrices to obtain convolution matrix
            cIdx = indices( k, : );    
            C = C_alpha + ...
                  C_gamma(cIdx(1),:)+C_gamma(cIdx(2),:)+C_gamma(cIdx(3),:);

            % compute the features 
            cLen = length(C)-2*padding;
            if padding1 == 0 || cLen<=0

                % without padding
                m1 = f1;
                m2 = m1+nMetrics-1;
                for f = 0:nFeatThisD-1

                    features( m1:m2, i) = ...
                        featureFcn( C, biases(b + f), inLen );

                    m1 = m2+1;
                    m2 = m2+nMetrics;

                end

            else
                % with padding
                m1 = f1;
                m2 = m1+nMetrics-1;
                for f = 0:nFeatThisD-1

                    features( m1:m2, i ) = ...
                        featureFcn( C(padding+1:end-padding), ...
                                      biases(b + f), cLen );
                    m1 = m2+1;
                    m2 = m2+nMetrics;

                end

            end
    
            f1 = f2 + 1;
            b = b + nFeatThisD;
    
        end
    
    end

end

% standardise by metric so all are in the same z-score range
fIdx = 0:nMetrics:f2-nMetrics;
for i = 1:nMetrics
    fIdx = fIdx + 1;
    features(fIdx, :) =  (features(fIdx, :)-mean(features(fIdx, :)))./...
                                std( features(fIdx, :) );
end

end



% the following functions have been optmised for fast execution
% wherever possible

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

% this is the slowest function by far
% the if statements are costly, as is the loop, but is unavoidable
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