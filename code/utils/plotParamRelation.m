function fig = plotParamRelation( x, y, ySD, ...
                                  xAxisLabel, ...
                                  yAxisLabel, ...
                                  legendNames, ...
                                  datasetNames, ...
                                  args )
    % Plot the parameter relationships from investigation reports
    arguments
        x                   {mustBeA( x, {'double', 'string', 'logical'})}
        y                   double
        ySD                 double
        xAxisLabel          string
        yAxisLabel          string
        legendNames         string
        datasetNames        string
        args.showLegend     logical = true
        args.showTitle      logical = true
        args.showXAxis      logical = true
        args.showYAxis      logical = true
        args.squarePlot     logical = false
        args.subPlotDim     double = []
        args.tileSpacing    string ...
            {mustBeMember( args.tileSpacing, ...
                {'Loose', 'Compact', 'Tight', 'None'} )} = 'Compact'
        args.legendPlot     string ...
            {mustBeMember( args.legendPlot, ...
                {'First', 'BottomMiddle'} )} = 'BottomMiddle'
    end
    
    [nModels, nValues, nDatasets] = size( y );

    % initialize the plot
    fig = figure;
    axes = gobjects( nDatasets, 1 );
    colours = lines( nModels );
    if isempty(args.subPlotDim)
        args.subPlotDim = [1 nDatasets];
    end

    tiledlayout( fig, args.subPlotDim(1), args.subPlotDim(2), ...
                 TileSpacing = args.tileSpacing );

    % plot the relations
    for i = 1:nDatasets

        axes(i) = nexttile;

        cla( axes(i) );
        hold( axes(i), 'on' );
        pltObj = gobjects( nModels, 1 );

        if args.squarePlot
            axes(i).PlotBoxAspectRatio = [1 1 1 ];
        end

        isBottomRow = (i > (args.subPlotDim(1)-1)*args.subPlotDim(2));
        isFirstCol = (mod( i, args.subPlotDim(2) ) == 1);
        isMiddleCol = (mod( i, args.subPlotDim(2) ) == ...
                                        floor(args.subPlotDim(2)/2)+1 );

        for j = 1:nModels

            xp = x + 0.01*(max(x)-min(x))*randn(1,nValues); 

            pltObj(j) = plot( axes(i), xp, y(j,:,i), ...
                                 LineWidth = 1, ...
                                 Color = colours(j,:), ...
                                 DisplayName = legendNames(j) );

            scatter( axes(i), xp, y(j,:,i), ...
                                 12, colours(j,:), "o", "Filled" );
            if ySD(j,:,i) > 0
                errorbar( axes(i), xp, y(j,:,i), ySD(j,:,i), ...
                          CapSize = 2, ... 
                          LineWidth = 1, Color=colours(j,:), ...
                          LineStyle= "none" );
            end

        end

        hold( axes(i), 'off' );

        % finalise the plot with formatting, etc
        if args.showTitle
            title( axes(i), datasetNames(i) );
        end

        if args.showLegend 
            switch args.legendPlot
                case 'First'
                    if i==1
                        legend( axes(i), pltObj, Location = 'best' );
                    end
                case 'BottomMiddle'
                    if isBottomRow && isMiddleCol
                        legendObj = legend( axes(i), pltObj, ...
                                            Location = 'south', ...
                                            Orientation = 'horizontal');
                        legendObj.Units = 'normalized';
                        legendObj.Position(2) = 0.02;
                        legendObj.Position(4) = 0.07;
                        legendObj.Box = false;
                    end
            end
        end
        
        if args.showXAxis && isBottomRow
            if isFirstCol
                xlabel( axes(i), xAxisLabel );
            end
            if isa( x, 'logical' )
                xlim( axes(i), [-0.5 1.5] );
                axes(i).XAxis.TickLabelsMode = 'manual';
                axes(i).XAxis.TickValues = [0 1];
                axes(i).XAxis.TickLabels = {'False' 'True'};
            end
        else
            axes(i).XAxis.TickLabels = [];
        end

        if args.showYAxis && isBottomRow && isFirstCol
            ylabel( axes(i), yAxisLabel );
        end

        if args.showYAxis
            setMinimalAxisTicks( axes(i), 'YAxis' );
            axes(i).YAxis.TickLabelFormat = '%.2f';
        else
            axes(i).YAxis.TickLabels = [];
        end            

        finalisePlot( axes(i) );

    end

end
