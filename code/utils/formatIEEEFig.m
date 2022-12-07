function fig = formatIEEEFig( fig, args )
% Prepare a figure for publication in IEEE journal
    arguments
        fig
        args.width              string ...
            {mustBeMember( args.width, ...
                {'Column', 'Page', ...
                 'Dataset', 'Components'} )} = 'Page'
        args.height             double = 1.50
        args.keepLegend         logical = true
        args.keepTitle          logical = true
        args.keepXAxisLabels    logical = true
        args.keepYAxisLabels    logical = true
        args.keepXAxisTicks     logical = true
        args.keepYAxisTicks     logical = true
        args.filename           string = []
    end

    fig.WindowStyle = 'normal';
    fig.Units = 'inches';

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
        if class(child) == "matlab.graphics.layout.TiledChartLayout"
            for j = 1:length(child.Children)
                updateGraphicsObject( child.Children(j), args );
            end
        else
            updateGraphicsObject( child, args );
        end

    end

    if ~isempty( args.filename )
        exportgraphics( fig, fullfile( path, args.filename ), ...
                        Resolution = 600, ContentType = "image" );
    end

end


function updateGraphicsObject( obj, args )

    obj.FontName = 'Times New Roman';
    obj.FontSize = 8;
    obj.Units = 'inches';

    switch class(obj)
        case "matlab.graphics.axis.Axes"

            obj.XAxis.TickLength = obj.XAxis.TickLength*4;

            if ~args.keepXAxisLabels
                obj.XAxis.Label = [];
            end

            if ~args.keepYAxisLabels
                obj.YAxis.Label = [];
            end
            obj.YAxis.Label.Position(1) = obj.YAxis.Label.Position(1)*1.05;

            if ~args.keepXAxisTicks
                obj.XAxis.TickValues = [];
            end

            if ~args.keepYAxisTicks
                obj.YAxis.TickValues = [];
            end

            if ~args.keepTitle
                obj.Title = [];
            end
      
        case 'matlab.graphics.illustration.Legend'
            obj.Visible = args.keepLegend;

    end


end
