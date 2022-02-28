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

function plotLatentComp( ax, decoder, Z, c, tSpan, fdPar )

nComp = size( Z, 1);
nPlots = min( nComp, length(ax) );

zScores = linspace( -2, 2, 3 );

% find the mean latent scores for centring 
Zmean = repmat( mean( Z, 2 ), 1, length(zScores) );

% ignore the null class
% nClass = nClass - 1;

for i = 1:nPlots
    
    % initialise component codes at their mean values
    ZComp = Zmean;

    % vary code i about its mean in a standardised way
    Zsd = std( Z(i,:) );
    for j = 1:length(zScores)
        ZComp(i,j) = Zmean(i,1) + zScores(j)*Zsd;
    end
    
    % duplicate for each class
    %dlZComp = dlarray( repmat( ZComp, 1, nClass ), 'CB' );
    dlZComp = dlarray( ZComp, 'CB' );

    % generate the curves using the decoder
    dlXComp = predict( decoder, dlZComp );
    XComp = double( extractdata( dlXComp ) );

    % select the requested channel
    XComp = squeeze( XComp(:,c,:) );

    % convert into smooth function
    XCompFd = smooth_basis( tSpan, XComp, fdPar );

    % plot the curves
    subplotFd( ax(i), XCompFd );

end


end