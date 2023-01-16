function [figs, axes]= initializePlots( XChannels, ZDim )
    % Setup plots for latent space and components
   
    % setup figure for Z distribution and clustering
    figs.LatentSpace = figure(1);
    clf;
    tiledlayout( figs.LatentSpace, 2, 2, TileSpacing = 'Compact' );
    axes.ZDistribution = nexttile;
    axes.ZClustering = nexttile;
    axes.AuxModel = nexttile;
    axes.AuxNetwork = nexttile;

    % setup the components figure
    figs.Components = figure(2);
    figs.Components.Position(3) = 100 + ZDim*250;
    figs.Components.Position(4) = 50 + XChannels*200;
    
    clf;
    tiledlayout( figs.Components, XChannels, ZDim, TileSpacing = 'Compact' );
    axes.Comp = gobjects( XChannels, ZDim );

    for i = 1:XChannels
        for j = 1:ZDim
            axes.Comp(i,j) = nexttile;
        end
    end

end