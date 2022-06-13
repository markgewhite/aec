function finalisePlot( ax, args )
    arguments
        ax
        args.square     logical = false
    end

    ax.Box = false;
    ax.TickDir = 'out';
    ax.XAxis.LineWidth = 1;
    ax.YAxis.LineWidth = 1;
    ax.FontName = 'Arial';
    if args.square
        ax.PlotBoxAspectRatio = [1 1 1];
    end

end