% ************************************************************************
% Function: plotLatentComp
%
% Plot characteristic curves of the latent codings which are similar
% in conception to the functional principal components
%
% Parameters:
%           ax          : array of axis objects
%           decoder     : trained decoder network
%           Z           : latent encodings sample
% ************************************************************************

function plotLatentComp( ax, decoder, Z, tSpan, fdPar )

nComp = size( Z, 1);
nPlots = min( nComp, length(ax) );

zScores = linspace( -2, 2, 3 );

mZ = repmat( mean( Z, 2 ), 1, length(zScores) );

for i = 1:nPlots
    
    % initialise component codes at their mean values
    cZ = mZ;

    % vary code i about its mean in a standardised way
    sdZ = std( Z(i,:) );
    for j = 1:length(zScores)
        cZ(i,j) = mZ(i,1) + zScores(j)*sdZ;
    end

    % generate the curves using the decoder
    dlcX = predict( decoder, dlarray( cZ, 'CB' ) );
    cX = double( extractdata( dlcX ) );

    % convert into smooth function
    cXFd = smooth_basis( tSpan, cX, fdPar );

    % plot the curves
    subplotFd( ax(i), cXFd );

end


end