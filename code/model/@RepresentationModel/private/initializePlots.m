function [figs, axes]= initializePlots( XChannels, ZDim, show )
    % Setup plots for latent space and components
    arguments
        XChannels       double
        ZDim            double
        show            logical = true
    end
   
    % setup figure for Z distribution and clustering
    figs.LatentSpace = getFigure( 1 );
    
    if show
        figs.LatentSpace.Visible = 'on';
    else
        figs.LatentSpace.Visible = 'off';
    end

    layout = tiledlayout( figs.LatentSpace, 2, 2, TileSpacing = 'Compact' );
    axes.ZDistribution = nexttile( layout );
    axes.ZClustering = nexttile( layout );
    axes.Input = nexttile( layout );
    axes.Output= nexttile( layout );

    % setup the components figure
    [figs.Components, axes.Comp] = initializeCompPlot( XChannels, ...
                                                       ZDim, ...
                                                       show );

end