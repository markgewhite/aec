function fig = formatElsevierFig( fig, args )
% Prepare a figure for publication in an Elsevier journal
    arguments
        fig
        args.sizeType           string ...
            {mustBeMember( args.sizeType, ...
                {'Minimal', 'SingleColumn', ...
                 '1.5Column', 'DoubleColumn', 'Custom'} )} = 'DoubleColumn'
        args.height             double = 4.0
        args.width              double = 0
        args.keepLegend         logical = true
        args.keepTitle          logical = true
        args.keepXAxisLabels    logical = true
        args.keepYAxisLabels    logical = true
        args.keepXAxisTicks     logical = true
        args.keepYAxisTicks     logical = true
        args.filename           string = []
    end

    fig.WindowStyle = 'normal';
    fig.Units = 'centimeters';

    switch args.sizeType
        case 'Minimal'
            dim = [ 3.0 args.height ];
        case 'SingleColumn'
            dim = [ 9.0 args.height ];
        case '1.5Column'
            dim = [ 14.0 args.height ];
        case 'DoubleColumn'
            dim = [ 19.0 args.height ];
        case 'Custom'
            dim = [ args.width args.height ];
    end

    fig.Position(3:4) = dim;   
    fig.Color = 'w';

    path = fileparts( which('utils/formatIEEEFig.m') );
    path = [path '/../../paper/figures/'];

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

    obj.FontName = 'Arial';
    obj.FontSize = 8;
    obj.Units = 'centimeters';

    switch class(obj)
        case "matlab.graphics.axis.Axes"

            obj.XAxis.TickLength = obj.XAxis.TickLength*4;

            if ~args.keepXAxisLabels
                obj.XAxis.Label = [];
            end
            obj.XAxis.Label.Position(1) = obj.XAxis.Label.Position(1)*0.90;

            if ~args.keepYAxisLabels
                obj.YAxis.Label = [];
            end
            obj.YAxis.Label.Position(1) = obj.YAxis.Label.Position(1)*0.90;

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
