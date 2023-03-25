function [figs, axes]= initializePlots( XChannels, ZDim, show )
    % Setup plots for latent space and components
    arguments
        XChannels       double
        ZDim            double
        show            logical = true
    end
   
    % setup figure for Z distribution and clustering
    figs.LatentSpace = figure(1);
    if show
        figs.LatentSpace.Visible = 'on';
    else
        figs.LatentSpace.Visible = 'off';
    end
    clf;
    tiledlayout( figs.LatentSpace, 2, 2, TileSpacing = 'Compact' );
    axes.ZDistribution = nexttile;
    axes.ZClustering = nexttile;
    axes.AuxModel = nexttile;
    axes.AuxNetwork = nexttile;

    % setup the components figure
    [figs.Components, axes.Comp] = initializeCompPlot( XChannels, ...
                                                       ZDim, ...
                                                       show );

end