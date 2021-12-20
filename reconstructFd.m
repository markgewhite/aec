% ************************************************************************
% Function: reconstructFd
%
% Reconstruct smooth curves from PCA functional data object
% using the harmonic FPCs, applying distict scores
% 
% Parameters:
%           pcaFd  : PCA functional data object
%           scores : loadings to apply to each of the harmonic functions
%           
% Outputs:
%           XFd    : reconstructed data objects
%
% ************************************************************************


function XFd = reconstructFd( pcaFd, scores, setup )

    % create a fine-grained time span from the existing basis
    basisFd = getbasis( pcaFd.meanfd );
    nKnots = getnbasis( basisFd );
    range = getbasisrange( basisFd );
    tFine = linspace( range(1), range(2), nKnots*10 );

    % create the set of points from the mean for each curve
    [ nCurves, nComp ] = size( scores );
    XPts = repmat( eval_fd( tFine, pcaFd.meanfd ), 1, nCurves );

    % linearly combine the components, pointswise
    for j = 1:nComp
        HPts = eval_fd( tFine, pcaFd.harmfd );
        for i = 1:nCurves
            XPts(:,i) = XPts(:,i) + scores(i,j)*HPts(:,j);
        end
    end

    % create the functional data object
    % (ought to be a better way than providing additional parameters)
    XFdPar = fdPar( basisFd, setup.penaltyOrder, setup.lambda );
    XFd = smooth_basis( tFine, XPts, XFdPar );

end