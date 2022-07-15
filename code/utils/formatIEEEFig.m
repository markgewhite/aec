function fig = formatIEEEFig( fig, args )
% Prepare a figure for publication in IEEE journal
    arguments
        fig
        args.width          string ...
            {mustBeMember( args.width, ...
                    {'Column', 'Page', 'Dataset', ...
                    'Components', 'Bespoke'} )} = 'Page'
        args.inches         double = 0
        args.size          string ...
            {mustBeMember( args.size, ...
                    {'Small', 'Medium', 'Large'} )} = 'Medium'
        args.keepLegend     logical = false
        args.keepTitle      logical = false
        args.keepAxisLabels logical = true
        args.keepAxisTicks  logical = true
        args.filename       string = []
    end

    fig.WindowStyle = 'normal';
    fig.Units = 'inches';

    
    switch args.size
        case 'Small'
            figHeight = 1.25;
            plotWidth = 0.75;
            plotHeight = 0.75;
        case 'Medium'
            figHeight = 1.75;
            plotWidth = 1.0;
            plotHeight = 1.0;
        case 'Large'
            figHeight = 2.00;
            plotWidth = 1.25;
            plotHeight = 1.25;
    end

    switch args.width
        case 'Column'
            dim = [ 3.45 figHeight ];
        case 'Page'
            dim = [ 7.11 figHeight ];
        case 'Dataset'
            dim = [ 1.8 figHeight ];
        case 'Components'
            dim = [ 6.01 figHeight ];
        case 'Bespoke'
            dim = [ args.inches figHeight ];
    end

    fig.Position(3:4) = dim;   
    fig.Color = 'w';

    path = fileparts( which('utils/formatIEEEFig.m') );
    path = [path '/../../paper/figures/'];

    j = 0;
    for i = 1:length(fig.Children)

        child = fig.Children(i);

        child.FontName = 'Times New Roman';
        child.FontSize = 8;
        child.Units = 'inches';

        switch class( child )
            case 'matlab.graphics.axis.Axes'
                
                plotCentre = [ child.InnerPosition(1)+0.5*child.InnerPosition(3) ...
                               child.InnerPosition(2)+0.5*child.InnerPosition(4) ];

                child.InnerPosition(1) = plotCentre(1)-0.5*plotWidth;
                child.InnerPosition(2) = plotCentre(2)-0.5*plotHeight;
                child.InnerPosition(3) = plotWidth;
                child.InnerPosition(4) = plotHeight;

                child.XAxis.LineWidth = 0.5;
                child.YAxis.LineWidth = 0.5;

                if ~args.keepAxisLabels
                    child.XAxis.Label = [];
                    child.YAxis.Label = [];
                end

                if args.keepAxisTicks
                    child.XAxis.TickValues = [ child.XAxis.TickValues(1), ...
                                               child.XAxis.TickValues(end) ];
                    child.YAxis.TickValues = [ child.YAxis.TickValues(1), ...
                                               child.YAxis.TickValues(end) ];
                else
                    child.XAxis.TickValues = [];
                    child.YAxis.TickValues = [];
                end

                if ~args.keepTitle
                    child.Title = [];
                end

           
            case 'matlab.graphics.illustration.Legend'
                child.Visible = args.keepLegend;


        end


    end

    if ~isempty( args.filename )
        exportgraphics( fig, fullfile( path, args.filename ), ...
                        Resolution = 600, ContentType = "image" );
    end

    
end
