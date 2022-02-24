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

function params = fitKernels( X, nFeatures, nMetrics, sampleRatio )

if iscell( X )
    % X is a cell array of time series of differing lengths
    % set a reference length
    inputLength = fix( max(cellfun( @length, X )) );
else
    % X is numeric array of time series of fixed length
    inputLength = size( X, 1 );
end
params.kernels = 84;
params.length = 9;
params.sampleRatio = sampleRatio;

if nMetrics >= 1 && nMetrics <=4
    params.metrics = nMetrics;
else
    error('Number of metrics not in correct range (1-4).');
end

maxDilationsPerKernel = 32;

% determine the dilations in proportion to time series length
[ params.dilations, params.featuresPerDilation ] = fitDilations( ...
                inputLength, params.kernels, nFeatures, ...
                params.length, maxDilationsPerKernel );

featuresPerKernel = sum( params.featuresPerDilation );

% compute quantile to fit within 95% CI of convolution distribution
quantiles = 0.025+0.95*quasiRandom( params.kernels*featuresPerKernel );

% compute biases based on convolution distributions
params.biases = fitBiases( X, params, quantiles );

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


function Q = quasiRandom( n )

    % low-discrepancy sequence to assign quantiles to kernel/dilation combinations
    Q = mod( (1:n)*(sqrt(5) + 1)/2, 1 );

end


function biases = fitBiases( X, parameters, Q  )

    % unpack parameters
    nKernels = parameters.kernels;
    kLen = parameters.length;
    dilations = parameters.dilations;
    nFeatPerD = parameters.featuresPerDilation;
    sampleRatio = parameters.sampleRatio;

    % determine X size
    lengthsVary = iscell( X );
    if lengthsVary
        nObs = length( X );
        XLens = cellfun( @length, X );
    else
        [ inLen, nObs ] = size( X );
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

    % sample from the training set
    nSample = fix( sampleRatio*nObs );

    nDil = length( dilations );

    nFeat = nKernels*sum( nFeatPerD );
    biases = zeros( nFeat, 1 );

    f1 = 1; % feature index counter: 1st in block of metrics

    % loop over all specified dilations
    for d = 1:nDil 

        % extract the specified dilation
        dilation = dilations( d );
        % set the padding in proportion
        padding = fix( ((kLen - 1) * dilation)/2 );

        % extract the number of features per dilation
        nFeatThisD = nFeatPerD( d ); 

        for k = 1:nKernels

            f2 = f1+ nFeatThisD - 1;

            % loop over randomly selected curve
            % to build up a representative convolution distribution

            % sample without replacement
            sample = randperm( nObs, nSample );
            % define long vector to hold multiple convolutions
            if lengthsVary
                C = zeros( sum( XLens(sample) ), 1 ); 
            else
                C = zeros( nSample*inLen, 1 ); 
            end

            c1 = 1;
            for i = 1:nSample

                % use a randomly selected curve
                if lengthsVary
                    x = X{ sample(i) };
                    inLen = XLens( sample(i) ); %costly?
                else
                    x = X( :, sample(i) );
                end
                c2 = c1 + inLen - 1;

                % obtain alpha and gamme vectors efficiently
                A = -x';        % A = alpha * X = -X
                G = x + x + x;  % G = gamma * X = 3X
    
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
        
                % combine alpha & gamma matrices to obtain convolution matrix
                cIdx = indices( k, : );    
                C( c1:c2 ) = C_alpha + ...
                      C_gamma(cIdx(1),:)+C_gamma(cIdx(2),:)+C_gamma(cIdx(3),:);

                c1 = c2 + 1;

            end

            % set biases from a pseudo-random quantile 
            % of the convolution distribution
            biases( f1:f2 ) = quantile( C, Q(f1:f2) );

            f1 = f2 + 1;

        end

    end

end



