function [ fig, axes ] = initializeCompPlot( XChannels, ZDim, show, asNew )
    % Setup the plot for latent components
    arguments
        XChannels       double
        ZDim            double
        show            logical = true
        asNew           logical = false
    end
   
    if asNew
        fig = figure;
    else
        fig = figure(2);
    end

    if show
        fig.Visible = 'on';
    else
        fig.Visible = 'off';
    end

    fig.Position(3) = 100 + ZDim*250;
    fig.Position(4) = 50 + XChannels*200;
    
    clf;
    tiledlayout( fig, XChannels, ZDim, TileSpacing = 'Compact' );
    axes = gobjects( XChannels, ZDim );

    for i = 1:XChannels
        for j = 1:ZDim
            axes(i,j) = nexttile;
        end
    end

end