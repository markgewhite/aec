function fig = formatIEEEFig( fig, args )
% Prepare a figure for publication in IEEE journal
    arguments
        fig
        args.width          string ...
            {mustBeMember( args.width, {'Column', 'Page'})} = 'Column'
        args.keepLegend     logical = false
        args.filename       string
    end

    fig.WindowStyle = 'normal';
    fig.Units = 'inches';

    switch args.width
        case 'Column'
            dim = [ 3.45 2.0 ];
        case 'Page'
            dim = [ 7.11 2.0 ];
    end

    fig.Position(3:4) = dim;   
    fig.Color = 'w';

    for i = 1:length(fig.Children)

        child = fig.Children(i);

        child.FontName = 'Times New Roman';
        child.FontSize = 8;
        
        switch class( child )
            case 'matlab.graphics.axis.Axes'
                child.InnerPosition(2) = 0.25;                
                child.InnerPosition(4) = 0.60;
                child.XAxis.LineWidth = 0.5;
                child.YAxis.LineWidth = 0.5;

                for j = 1:length(child.Children)
                    grandchild = child.Children(j);
                    switch class( grandchild )
                        case 'matlab.graphics.chart.primitive.Line'
                            if isempty( grandchild.DisplayName )
                                grandchild.LineWidth = 0.25;
                            else
                                grandchild.LineWidth = 0.5;
                            end
                                
                    end
                end

            case 'matlab.graphics.illustration.Legend'
                child.Visible = args.keepLegend;


        end

    end

    if ~isempty( args.filename )
        path = fileparts( which('utils/formatIEEEFig.m') );
        path = [path '/../../figures/'];
        exportgraphics( fig, fullfile( path, args.filename ), ...
                        Resolution = 600, ContentType = "image" );
    end

    
end
