function finalisePlot( ax, args )
    arguments
        ax
        args.square         logical = false
        args.minimalTicks   logical = false
    end

    ax.Box = false;
    ax.TickDir = 'out';
    ax.XAxis.LineWidth = 0.75;
    ax.YAxis.LineWidth = 0.75;
    ax.FontName = 'Arial';
    ax.FontSize = 8;
    %ax.TickLabelInterpreter = 'latex';
    %ax.XLabel.Interpreter = 'latex';
    %ax.YLabel.Interpreter = 'latex';
    %ax.Title.Interpreter = 'latex';
    ax.Title.VerticalAlignment = 'baseline';
    
    if args.square
        ax.PlotBoxAspectRatio = [1 1 1];
    end

    if args.minimalTicks
        setMinimalAxisTicks( ax, 'XAxis' );
        setMinimalAxisTicks( ax, 'YAxis' );
    end

end