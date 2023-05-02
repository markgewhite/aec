function [figs, axes]= initializePlots( XChannels, ZDim, show )
    % Setup plots for latent space and components
    arguments
        XChannels       double
        ZDim            double
        show            logical = true
    end
   
    % setup figure for Z distribution and clustering
    allFigs = findall( groot, Type = 'Figure');
    if isempty(allFigs)
        figs.LatentSpace = figure(1);
    else
        figs.LatentSpace = allFigs(1);
        clf( figs.LatentSpace );
    end
    
    if show
        figs.LatentSpace.Visible = 'on';
    else
        figs.LatentSpace.Visible = 'off';
    end

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