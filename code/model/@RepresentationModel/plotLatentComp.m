function plotLatentComp( self, args )
    % Plot characteristic curves of the latent codings which are similar
    % in conception to the functional principal components
    arguments
        self                RepresentationModel
        args.XMean          {mustBeA( args.XMean, { 'dlarray', 'double' })} = []
        args.XC             {mustBeA( args.XC, { 'dlarray', 'double' })} = []
        args.nSample        double = 0
        args.order          double ...
            {mustBeInteger, mustBePositive} = []
        args.type           char ...
            {mustBeMember(args.type, ...
                {'Smoothed', 'Predicted', 'Both'} )} = 'Smoothed'
        args.shading        logical = true
        args.showLegend     logical = true
        args.showTitle      logical = true
        args.showXAxis      logical = true
        args.showYAxis      logical = true
        args.centredYAxis   logical = false
        args.axes           = []
    end
    
    if isempty( args.XMean )
        % use the pre-calculated mean curve
        XMean = repmat( self.MeanCurve, 1, self.ZDim );
        if self.XChannels==1
            XMean = permute( XMean, [1 3 2] );
        end
        tSpanMean = self.TSpan.Regular;
    else
        % use the mean curve specified
        if isa( args.XMean, 'dlarray' )
            XMean = double( extractdata( args.XMean ) );
        else
            XMean = args.XMean;
        end
        switch size( XMean, 1 )
            case length(self.TSpan.Regular)
                tSpanMean = self.TSpan.Regular;
            case length(self.TSpan.Target)
                tSpanMean = self.TSpan.Target;
            otherwise
                tSpanMean = linspace( self.TSpan.Original(1), ...
                                      self.TSpan.Original(end), length(XMean) );
        end
    end

    if isempty( args.XC )
        % use the pre-calculated latent components
        XC = self.LatentComponents;
        tSpanXC = self.TSpan.Regular;

    else
        % use the latent components specified
        if isa( args.XC, 'dlarray' )
            XC = double( extractdata( args.XC ) );
        else
            XC = args.XC;
        end
        switch size( XC, 1 )
            case length(self.TSpan.Regular)
                tSpanXC = self.TSpan.Regular;
            case length(self.TSpan.Target)
                tSpanXC = self.TSpan.Target;
            otherwise
                tSpanXC = linspace( self.TSpan.Original(1), ...
                                    self.TSpan.Original(end), length(XMean) );
        end
    end

    % get dimensions
    [nPts, nSamples, nDim, nChannels, ] = size( XC );

    % interpolate all curves over the plot's points
    tSpanPlot = linspace( self.TSpan.Original(1), ...
                          self.TSpan.Original(end), 101 );

    XMeanPlot = zeros( 101, nDim, nChannels );
    XCPlot = zeros( 101, nSamples, nDim, nChannels ); 
    for c = 1:nChannels
        for d = 1:nDim
            XMeanPlot(:,d,c) = interp1( tSpanMean, XMean(:,:,d,c), tSpanPlot );
            for s = 1:nSamples
                XCPlot(:,s,d,c) = XMeanPlot(:,d,c)' + ...
                                    interp1( tSpanXC, XC(:,s,d,c), tSpanPlot );
            end
        end
    end

    % set the colours: red=positive, blue=negative
    compColours = [ 0.0000 0.4470 0.7410; ...
                    0.6350 0.0780 0.1840 ];
    gray = [ 0.5, 0.5, 0.5 ];
    black = [ 0, 0 , 0 ];
    % set positive/negative plot characteristics
    names = [ "-ve", "+ve" ];

    % set the plot axes
    if isempty( args.axes )
        axes = self.Axes.Comp;
    else
        axesDim = size( args.axes );
        if axesDim(1) == self.XChannels ...
                && axesDim(2) == self.ZDim
            axes = args.axes;
        else
            eid = 'Plot:AxesDimsIncorrect';
            msg = 'The specified axes array does not have correct dimensions.';
            throwAsCaller( MException(eid,msg) );
        end
    end

    if isempty( args.order )
        % standard order from the model
        compIdx = 1:nDim;
    elseif length(args.order)==nDim ...
           && max(args.order)<=nDim
        % specified order
        compIdx = args.order;
    else
        eid = 'Plot:InvalidOrder';
        msg = 'The specified component order is invalid.';
        throwAsCaller( MException(eid,msg) );
    end

    % set shading transparency
    alpha = max( 0.025*(18-nSamples), 0.10 );

    for c = 1:nChannels

        for i = compIdx

            axis = axes(c,i);

            cla( axis );
            hold( axis, 'on' );

            pltObj = gobjects( 3, 1 );

            l = 0; % legend counter

            % plot the gradations (in reverse to +ve plots first)
            for j = nSamples:-1:1

                % first half blue (-ve), second half red (+ve)
                s = (j > 0.5*nSamples) + 1;

                isOuterCurve = (j==1 || j==nSamples);
                if isOuterCurve
                    width = 1.0;               
                else
                    width = 0.5;
                end

                % fill-up shading area
                if args.shading
                    plotShadedArea( axis, ...
                                    tSpanPlot, ...
                                    XCPlot( :,j,i,c ), ...
                                    XMeanPlot( :,i,c ), ...
                                    compColours( s,: ), ...
                                    name = names(s), ...
                                    alpha = alpha );
                end

                % plot gradation curve

                if any(strcmp( args.type, {'Predicted','Both'} ))
                    % plot predicted values
                    plot( axis, ...
                          fda.tSpanAdaptive, XC( :,j,i,c ), ...
                          Color = gray, ...
                          LineWidth = 0.5 );
                end
                if any(strcmp( args.type, {'Smoothed','Both'} ))
                    % plot smoothed curves
                    if isOuterCurve
                        % include in legend
                        l = l+1;
                        pltObj(l) = plot( axis, ...
                                          tSpanPlot, XCPlot( :,j,i,c ), ...
                                          Color = compColours( s,: ), ...
                                          LineWidth = width, ...
                                          DisplayName = names(s));
                    else
                        % don't record it for the legend
                        plot( axis, ...
                                          tSpanPlot, XCPlot( :,j,i,c ), ...
                                          Color = compColours( s,: ), ...
                                          LineWidth = width );
                    end

                end

            end

            % plot the mean curve
            l = l+1;
            pltObj(l) = plot( axis, ...
                              tSpanPlot, XMeanPlot( :,i,c ), ...
                              Color = black, ...
                              LineWidth = 1.5, ...
                              DisplayName = 'Mean' );

            hold( axis, 'off' );

            % finalise the plot with formatting, etc
            if args.showLegend && c==1 && i==1
                legend( axis, pltObj, Location = 'best' );
            end
            
            if args.showTitle && c==1
                title( axis, ['Component ' num2str(i)] );
            end
            
            if args.showXAxis && c==self.XChannels
                xlabel( axis, self.Info.TimeLabel );
                xlim( axis, [tSpanPlot(1) tSpanPlot(end)] );
            else
                axis.XAxis.TickLabels = [];
            end

            if args.showYAxis && i==1
                ylabel( axis, self.Info.ChannelLabels(c) );
                axis.YAxis.TickLabelFormat = '%.1f';
            else
                axis.YAxis.TickLabels = [];
            end

            if ~isempty( self.Info.ChannelLimits )
                if args.centredYAxis
                    yRng = self.Info.ChannelLimits(c,2) ...
                            - self.Info.ChannelLimits(c,1);
                    yLimits = [ -0.5*yRng, 0.5*yRng ];
                else
                    yLimits = self.Info.ChannelLimits(c,:);
                end
                ylim( axis, yLimits );
            end               

            finalisePlot( axis );

        end

    end

end
