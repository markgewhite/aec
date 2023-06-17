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
    figs.LatentSpace.Position(4) = 250;

    layout = tiledlayout( figs.LatentSpace, 1, 2, TileSpacing = 'Compact' );
    axes.ZDistribution = nexttile( layout );
    axes.ZClustering = nexttile( layout );

    % setup the components figure
    [figs.Components, axes.Comp] = initializeCompPlot( XChannels, ...
                                                       ZDim, ...
                                                       show );

    % setup the figure for predictions plots
    figs.Predictions = getFigure( 4 );    
    if show
        figs.Predictions.Visible = 'on';
    else
        figs.Predictions.Visible = 'off';
    end
    figs.Predictions.Position(3) = 450;
    figs.Predictions.Position(4) = 50 + XChannels*300;
    
    layout = tiledlayout( figs.Predictions, XChannels, 1, TileSpacing = 'Compact' );
    axes.Pred = gobjects( XChannels, 1 );
    for c = 1:XChannels
        axes.Pred(c) = nexttile( layout );
    end   

end