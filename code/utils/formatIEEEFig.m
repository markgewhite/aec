function fig = formatIEEEFig( fig, args )
% Prepare a figure for publication in IEEE journal
    arguments
        fig
        args.width          string ...
            {mustBeMember( args.width, {'Column', 'Page', ...
                    'Dataset', 'Components', 'Bespoke'} )} = 'Column'
        args.inches         double = 0
        args.keepLegend     logical = false
        args.filename       string = []
    end

    fig.WindowStyle = 'normal';
    fig.Units = 'inches';

    height = 1.75;
    switch args.width
        case 'Column'
            dim = [ 3.45 height ];
        case 'Page'
            dim = [ 7.11 height ];
        case 'Dataset'
            dim = [ 1.8 height ];
        case 'Components'
            dim = [ 6.01 height ];
        case 'Bespoke'
            dim = [ args.inches 2.0 ];
    end

    set(gca,'LooseInset',get(gca,'TightInset'));
    fig.Position(3:4) = dim;   
    fig.Color = 'w';

    plotWidth = 1.0;
    plotHeight = 1.0;

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

                child.XAxis.TickValues = [ child.XAxis.TickValues(1), ...
                                           child.XAxis.TickValues(end) ];
                child.YAxis.TickValues = [ child.YAxis.TickValues(1), ...
                                           child.YAxis.TickValues(end) ];
           
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
