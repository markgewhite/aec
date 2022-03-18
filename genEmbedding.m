% ************************************************************************
% Function: genEmbedding
%
% Generate the embedding using the Rocket transform
% optionally with PCA applied
%
% Parameters:
%           XTrn     : training data
%           XTst     : testing data
%           setup    : embedding setup
% Outputs:
%           X        : data as cell array, variable length (fine)
%           XN       : data as numeric array, normalised length (coarse)
%
% ************************************************************************

function [ XTTrn, XTTst, params ] = genEmbedding( XTrn, XTst, setup )

% fit the kernels using the training data
params = fitKernels( XTrn, setup.nKernels, ...
                     setup.nMetrics, setup.sampleRatio );

% perform the transform
XTTrn = rocketTransform( XTrn, params );
XTTst = rocketTransform( XTst, params );

% reduce with PCA, if required
if setup.usePCA

    % perform PCA on the training data: input is N x P, so transpose 
    % (works on the centred data)
    % turn off warning about linear dependence
    warning( 'off', 'stats:pca:ColRankDefX' );
    [pcaCoeff, XTTrnPCA, ~, ~, pcaExplained, XTMean ] = pca( XTTrn' );
    warning( 'on', 'stats:pca:ColRankDefX' );

    % apply loadings to the centred test data (transpose implied)
    XTTstPCA = (XTTst-XTMean')'*pcaCoeff;

    % determine how many components to retain
    nRetained = sum( pcaExplained > setup.retainThreshold );

    % truncate to retained components (back transpose for training)
    XTTrn = XTTrnPCA( :, 1:nRetained )';
    XTTst = XTTstPCA( :, 1:nRetained )';

end

end