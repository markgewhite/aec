function fig = plot( self, args )
    % Plot the curves of a given data set altogether,
    % revealing the classes
    arguments
        self                ModelDataset
        args.nSample        double = 250
        args.showLegend     logical = true
        args.showTitle      logical = true
        args.showXAxis      logical = true
        args.showYAxis      logical = true
    end
    
    if args.nSample > 0
        nSample = min( args.nSample, self.NumObs );
    else
        nSample = self.NumObs;
    end

    % select the observations/curves to plot
    curves = randsample( self.NumObs, nSample )';

    % initialize the plot
    fig = figure(4);
    tiledlayout( fig, self.XChannels, 1, TileSpacing = 'Compact' );
    axes = gobjects( self.XChannels, 1 );
    pltObj = gobjects( self.CDim, 1 );
    for c = 1:self.XChannels
        axes(c) = nexttile;
        cla( axes(c) );
        hold( axes(c), 'on' );
    end

    colours = lines( 9 );
    colours = colours( 3:9, : );

    % plot the curves
    t = self.TSpan.Regular;
    X = self.XInputRegular;
    Y = self.Y;
    classInLegend = false( self.CDim, 1 );

    for i = curves

        % prepare the colour based on class with random saturation
        hsv = rgb2hsv( colours( Y(i), : ) );
        hsv(2) = 0.5+0.5*rand();
        rgb = hsv2rgb( hsv );

        for c = 1:self.XChannels

            if classInLegend( Y(i) )
                % plot the curve straight
                plot( axes(c), ...
                      t, X( :, i, c ), ...
                      LineWidth = 1, ...
                      Color = rgb );
            else
                % plot the curve and update the legend
                classInLegend( Y(i) ) = true;
                classLabel = self.Info.ClassLabels( Y(i) );
                pltObj( Y(i) ) = plot( axes(c), ...
                                  t, X( :, i, c ), ...
                                  Color = rgb, ...
                                  LineWidth = 1, ...
                                  DisplayName = classLabel );

            end

        end

    end

    for c = 1:self.XChannels

        hold( axes(c), 'off' );

        % finalise the plot with formatting, etc
        if args.showLegend && c==1
            legend( axes(c), pltObj, Location = 'best' );
        end
        
        if args.showXAxis && c==self.XChannels
            xlabel( axes(c), self.Info.TimeLabel );
            xlim( axes(c), [t(1) t(end)] );
        else
            axes(c).XAxis.TickLabels = [];
        end

        if args.showYAxis 
            ylabel( axes(c), self.Info.ChannelLabels(c) );
            axes(c).YAxis.TickLabelFormat = '%.1f';
        else
            axes(c).YAxis.TickLabels = [];
        end

        if ~isempty( self.Info.ChannelLimits )
            ylim( axes(c), self.Info.ChannelLimits(c,:) );
        end               

        finalisePlot( axes(c), minimalTicks = true );

    end

    if self.XChannels==1
        axes.PlotBoxAspectRatio = [1 1 1 ];
    end

end
