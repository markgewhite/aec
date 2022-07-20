function [figs, axes]= initializePlots( XChannels, ZDim )
    % Setup plots for latent space and components
   
    % setup figure for Z distribution and clustering
    figs.LatentSpace = figure(1);
    clf;
    axes.ZDistribution = subplot( 1, 2, 1 );
    axes.ZClustering = subplot( 1, 2, 2 );

    % setup the components figure
    figs.Components = figure(2);
    figs.Components.Position(3) = 100 + ZDim*250;
    figs.Components.Position(4) = 50 + XChannels*200;
    
    clf;
    axes.Comp = gobjects( XChannels, ZDim );

    for i = 1:XChannels
        for j = 1:ZDim
            axes.Comp(i,j) = subplot( XChannels, ZDim, (i-1)*ZDim + j );
        end
    end

end