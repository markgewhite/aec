function fig = plotParamRelation( report, paramName, metric, metricName, ...
                                  datasetNames, legendNames, args )
    % Plot the parameter relationships from investigation reports
    arguments
        report              struct
        paramName           string
        metric              string
        metricName          string
        datasetNames        string
        legendNames         string
        args.resultSet      string = "TestingResults"
        args.showLegend     logical = true
        args.showTitle      logical = true
        args.showXAxis      logical = true
        args.showYAxis      logical = true
        args.subPlotDim     double = []
    end
    
    nDim = length( report.GridSearch );
    nModels = length( report.GridSearch{1} );
    nValues = length( report.GridSearch{2} );
    nDatasets = length( report.GridSearch{3} );

    x = report.GridSearch{2};
    y = report.(args.resultSet).Mean.(metric);
    ySD = report.(args.resultSet).SD.(metric);

    % initialize the plot
    fig = figure;
    axes = gobjects( nDatasets, 1 );
    colours = lines( 3 );
    if isempty(args.subPlotDim)
        args.subPlotDim = [1 nDatasets];
    end

    % plot the relations
    for i = 1:nDatasets

        axes(i) = subplot( args.subPlotDim(1), ...
                           args.subPlotDim(2), i );
        cla( axes(i) );
        hold( axes(i), 'on' );
        pltObj = gobjects( nModels, 1 );

        for j = 1:nModels

            xp = x + 0.01*(max(x)-min(x))*randn(1,nValues); 
            pltObj(j) = scatter( axes(i), xp, y(j,:,i), ...
                                 10, colours(j,:), "o", "Filled", ...
                                 DisplayName = legendNames(j) );
            errorbar( axes(i), xp, y(j,:,i), ySD(j,:,i), ...
                      CapSize = 2, ... 
                      LineWidth = 1, Color=colours(j,:), ...
                      LineStyle= "none" );

        end

        hold( axes(i), 'off' );

        % finalise the plot with formatting, etc
        if args.showTitle
            title( axes(i), datasetNames(i) );
        end

        if args.showLegend && i==1
            legend( axes(i), pltObj, Location = 'best' );
        end
        
        if args.showXAxis
            if i == 1
                xlabel( axes(i), paramName );
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

        if args.showYAxis
            if i==1
                ylabel( axes(i), metricName );
            end
            ymin = round( axes(i).YAxis.Limits(1), 2 );
            ymax = round( axes(i).YAxis.Limits(2), 2 );
            if ymin == 0
                ymin = axes(i).YAxis.Limits(1);
            end
            if ymax == 0
                ymax = axes(i).YAxis.Limits(2);
            end
            if 2*round(ymax/2, 2)~=ymax
                ymax = ymax+0.01;
            end
            if 2*round(ymin/2, 2)~=ymin
                ymax = ymax-0.01;
            end
            if ymin >= 0
                ylim( axes(i), [0 ymax] );
                axes(i).YAxis.TickValues = [0, ymax/2, ymax];
            else
                ylim( axes(i), [ymin ymax] );
                axes(i).YAxis.TickValues = [ymin, 0, ymax];
            end
            axes(i).YAxis.TickLabelFormat = '%.2f';

        else
            axes(i).YAxis.TickLabels = [];
        end            

        finalisePlot( axes(i) );

    end

end
