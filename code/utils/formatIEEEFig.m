function fig = formatIEEEFig( fig, args )
% Prepare a figure for publication in IEEE journal
    arguments
        fig
        args.width              string ...
            {mustBeMember( args.width, ...
                {'Column', 'Page', ...
                 'Dataset', 'Components'} )} = 'Page'
        args.height             double = 1.50
        args.size               string ...
            {mustBeMember( args.size, ...
                    {'Small', 'Medium', 'Large'} )} = 'Medium'
        args.keepLegend         logical = false
        args.keepTitle          logical = false
        args.keepXAxisLabels    logical = true
        args.keepYAxisLabels    logical = true
        args.keepXAxisTicks     logical = true
        args.keepYAxisTicks     logical = true
        args.filename           string = []
    end

    fig.WindowStyle = 'normal';
    fig.Units = 'inches';

    
    switch args.size
        case 'Small'
            plotWidth = 0.50;
            plotHeight = 0.50;
            margin = 0.05;
        case 'Medium'
            plotWidth = 0.75;
            plotHeight = 0.75;
            margin = 0.10;
        case 'Large'
            plotWidth = 1.00;
            plotHeight = 1.00;
            margin = 0.15;
    end

    switch args.width
        case 'Column'
            dim = [ 3.45 args.height ];
        case 'Page'
            dim = [ 7.11 args.height ];
        case 'Dataset'
            dim = [ 1.8 args.height ];
        case 'Components'
            dim = [ 6.01 args.height ];
    end

    fig.Position(3:4) = dim;   
    fig.Color = 'w';

    path = fileparts( which('utils/formatIEEEFig.m') );
    path = [path '/../../paper/figures/'];

    p = 0;
    for i = 1:length(fig.Children)

        child = fig.Children(i);

        child.FontName = 'Times New Roman';
        child.FontSize = 8;
        child.Units = 'inches';

        switch class( child )
            case 'matlab.graphics.axis.Axes'

                child.XAxis.TickLength = child.XAxis.TickLength*4;

                if ~args.keepXAxisLabels
                    child.XAxis.Label = [];
                end

                if ~args.keepYAxisLabels
                    child.YAxis.Label = [];
                end
                child.YAxis.Label.Position(1) = child.YAxis.Label.Position(1)*1.05;

                if ~args.keepXAxisTicks
                    child.XAxis.TickValues = [];
                end

                if ~args.keepYAxisTicks
                    child.YAxis.TickValues = [];
                end

                if ~args.keepTitle
                    child.Title = [];
                end

                p = p + 1;
                plotIdx(p) = i; %#ok<AGROW> 
                plotCentre(p,:) = [ child.InnerPosition(1)+0.5*child.InnerPosition(3) ...
                               child.InnerPosition(2)+0.5*child.InnerPosition(4) ]; %#ok<AGROW> 

          
            case 'matlab.graphics.illustration.Legend'
                child.Visible = args.keepLegend;

        end


    end

    spacing = (fig.Position(3)-2*margin)/p;
    for i = 1:p

        child = fig.Children( plotIdx(i) );

        %child.InnerPosition(1) = (p-i+0.5)*spacing - 0.5*margin;
        %child.InnerPosition(2) = plotCentre(p,2)-0.5*plotHeight;
        child.InnerPosition(3) = plotWidth;
        child.InnerPosition(4) = plotHeight;

    end


    if ~isempty( args.filename )
        exportgraphics( fig, fullfile( path, args.filename ), ...
                        Resolution = 600, ContentType = "image" );
    end

    
end
