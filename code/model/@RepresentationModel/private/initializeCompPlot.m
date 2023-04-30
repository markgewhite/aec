function [ fig, axes ] = initializeCompPlot( XChannels, ZDim, show )
    % Setup the plot for latent components
    arguments
        XChannels       double
        ZDim            double
        show            logical = true
    end
   
    allFigs = findall( groot, Type = 'Figure');
    if length(allFigs)==2
        fig = figure(2);
    else
        fig = allFigs(2);
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