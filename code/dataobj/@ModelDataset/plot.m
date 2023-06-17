function fig = plot( self, args )
    % Plot the curves of a given data set altogether,
    % revealing the classes
    arguments
        self                ModelDataset
        args.nSample        double = 500
        args.tSpan          double = []
        args.X              double = []
        args.Y              double = []
        args.axes
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

    if isfield( args, 'axes' )
        % use the provided array of axes
        axes = args.axes;
        if length( axes )~= self.XChannels
            eid = 'ModelDataset:AxesInvalid';
            msg = 'The number of axes do not match the number of channels.';
            throwAsCaller( MException(eid,msg) );
        end

    else
        % initialize the plot
        fig = getFigure( 4 );
        layout = tiledlayout( fig, self.XChannels, 1, TileSpacing = 'Compact' );
        axes = gobjects( self.XChannels, 1 );
        for c = 1:self.XChannels
            axes(c) = nexttile( layout );
        end
    end
    pltObj = gobjects( self.CDim, 1 );

    % get the data, either from self or as specified
    if isempty( args.tSpan )
        t = self.TSpan.Input;
    else
        t = args.tSpan;
    end
    if isempty( args.X )
        X = self.XInput;
    else
        X = args.X;
    end
    if isempty( args.Y )
        Y = self.Y;
    else
        Y = args.Y;
    end

    % set the colour scheme
    if self.CDim==0
        % no classes, only a continuous range
        cmap = jet(self.NumObs);
        
        % map Y values to the colormap
        % first normalize Y between 1 and NumObs
        YNorm = round( (Y - min(Y(:))) / (max(Y(:)) - min(Y(:))) * (self.NumObs-1) + 1 );
        
        % select colour for each curve based on Y
        colours = cmap( YNorm, : );

        classInLegend = true;
        C = ones( length(Y) );
    else
        colours = lines( self.CDim );
        classInLegend = false( self.CDim, 1 );
        [~, ~, C] = unique( Y );
    end

    % iterate over the channels
    for c = 1:self.XChannels

        cla( axes(c) );
        hold( axes(c), 'on' );
    
        % plot the curves
        for i = curves
    
            % prepare the colour based on class with random saturation
            if self.CDim==0
                rgb = colours( i, : );
            else
                hsv = rgb2hsv( colours( C(i), : ) );
                hsv(2) = 0.5+0.5*rand();
                rgb = hsv2rgb( hsv );
            end
    
            if classInLegend( C(i) )
                % plot the curve straight
                plot( axes(c), ...
                      t, X( :, i, c ), ...
                      LineWidth = 1, ...
                      Color = rgb );
            else
                % plot the curve and update the legend
                classLabel = self.Info.ClassLabels( C(i) );
                classInLegend( C(i) ) = true;
                pltObj( C(i) ) = plot( axes(c), ...
                                       t, X( :, i, c ), ...
                                       Color = rgb, ...
                                       LineWidth = 1, ...
                                       DisplayName = classLabel );

            end
    
        end

        hold( axes(c), 'off' );

        % finalise the plot with formatting, etc
        if args.showLegend && c==1
            if self.CDim==0
                colormap( axes(c), cmap );
                cbar = colorbar( axes(c) );
                cbar.Label.String = self.Info.ClassLabels;
                clim( axes(c), [min(Y(:)), max(Y(:))] );

            else
                if all(ishandle( pltObj ))
                    legend( axes(c), pltObj, Location = 'best' );
                end
            end
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
