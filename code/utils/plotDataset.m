function fig = plotDataset( thisData, args )
    % Plot the curves of a given data set altogether,
    % revealing the classes
    arguments
        thisData            ModelDataset
        args.nSample        double = 0
        args.showLegend     logical = true
        args.showTitle      logical = true
        args.showXAxis      logical = true
        args.showYAxis      logical = true
    end
    
    if args.nSample > 0
        nSample = min( args.nSample, thisData.NumObs );
    else
        nSample = thisData.NumObs;
    end

    % select the observations/curves to plot
    curves = randsample( thisData.NumObs, nSample )';

    % smooth and re-evaluate all curves
    tSpanPlot = linspace( thisData.TSpan.Original(1), ...
                          thisData.TSpan.Original(end), 101 );

    % initialize the plot
    fig = figure;
    axes = gobjects( thisData.XChannels, 1 );
    pltObj = gobjects( thisData.CDim, 1 );
    for c = 1:thisData.XChannels
        axes(c) = subplot( 1, thisData.XChannels, c );
        cla( axes(c) );
        hold( axes(c), 'on' );
    end

    colours = lines( 9 );
    colours = colours( 3:9, : );

    % plot the curves
    X = thisData.XTarget;
    Y = thisData.Y;
    classInLegend = false( thisData.CDim, 1 );

    for i = curves

        % prepare the colour based on class with random saturation
        hsv = rgb2hsv( colours( Y(i), : ) );
        hsv(2) = 0.5+0.5*rand();
        rgb = hsv2rgb( hsv );

        for c = 1:thisData.XChannels

            if classInLegend( Y(i) )
                % plot the curve straight
                plot( axes(c), ...
                      tSpanPlot, X( :, i, c ), ...
                      LineWidth = 0.5, ...
                      Color = rgb );
            else
                % plot the curve and update the legend
                classInLegend( Y(i) ) = true;
                classLabel = thisData.Info.ClassLabels( Y(i) );
                pltObj( Y(i) ) = plot( axes(c), ...
                                  tSpanPlot, X( :, i, c ), ...
                                  Color = rgb, ...
                                  LineWidth = 0.5, ...
                                  DisplayName = classLabel );

            end

        end

    end

    for c = 1:thisData.XChannels

        hold( axes(c), 'off' );

        % finalise the plot with formatting, etc
        if args.showLegend && c==1
            legend( axes(c), pltObj, Location = 'best' );
        end
        
        if args.showXAxis
            xlabel( axes(c), thisData.Info.TimeLabel );
            xlim( axes(c), [tSpanPlot(1) tSpanPlot(end)] );
        else
            axes(c).XAxis.TickLabels = [];
        end

        if args.showYAxis && c==1
            ylabel( axes(c), thisData.Info.ChannelLabels(c) );
            axes(c).YAxis.TickLabelFormat = '%.1f';
        else
            axes(c).YAxis.TickLabels = [];
        end

        if ~isempty( thisData.Info.ChannelLimits )
            ylim( axes(c), thisData.Info.ChannelLimits(c,:) );
        end               

        finalisePlot( axes(c) );

    end

end
