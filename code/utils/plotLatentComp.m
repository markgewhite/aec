function plotLatentComp( thisModel, args )
    % Plot characteristic curves of the latent codings which are similar
    % in conception to the functional principal components
    arguments
        thisModel           {mustBeA( thisModel, ...
                                { 'FullRepresentationModel', ...
                                  'CompactRepresentationModel' })}
        args.XC             {mustBeA( args.XC, { 'dlarray', 'double' })} = []
        args.nSample        double = 0
        args.type           char ...
            {mustBeMember(args.type, ...
                {'Smoothed', 'Predicted', 'Both'} )} = 'Smoothed'
        args.shading        logical = false
        args.legend         logical = true
    end
    
    if isempty( args.XC )
        % use the pre-calculated latent components
        XC = thisModel.LatentComponents;
        % set the appropriate timespans and FD parameters
        tSpanXC = thisModel.TSpan.Regular;
        fdParams = thisModel.FDA.FdParamsRegular;

    else
        % use the latent components specified
        if isa( args.XC, 'dlarray' )
            XC = double( extractdata( args.XC ) );
        else
            XC = args.XC;
        end
        % set the appropriate timespans and FD parameters
        tSpanXC = thisModel.TSpan.Target;
        fdParams = thisModel.FDA.FdParamsTarget;
        
    end

    % re-order the dimensions for FDA
    if size( XC, 3 ) > 1
        XC = permute( XC, [1 3 2] );
    end

    % smooth and re-evaluate all curves
    tSpanPlot = linspace( thisModel.TSpan.Original(1), ...
                          thisModel.TSpan.Original(end), 101 );
    XCFd = smooth_basis( tSpanXC, XC, fdParams );
    XCsmth = eval_fd( tSpanPlot, XCFd );

    % set the colours from blue and red
    compColours = [ 0.0000 0.4470 0.7410; ...
                    0.6350 0.0780 0.1840 ];
    gray = [ 0.5, 0.5, 0.5 ];
    black = [ 0, 0 , 0 ];
    % set positive/negative plot characteristics
    names = [ "+ve", "-ve" ];

    if args.nSample > 0
        nSample = args.nSample;
    else
        nSample = thisModel.NumCompLines;
    end

    for c = 1:thisModel.XChannels

        k = 0; % sample counter

        for i = 1:thisModel.ZDim

            axis = thisModel.Axes.Comp(c,i);

            cla( axis );
            hold( axis, 'on' );

            pltObj = gobjects( 3, 1 );

            l = 0; % legend counter

            % plot the gradations
            for j = 1:nSample

                % set the sign/colour counter
                s = round( j/nSample, 0 ) + 1;
                
                % next sample
                k = k+1;

                isOuterCurve = (j==1 || j==nSample);
                if isOuterCurve
                    % fill-up shading area
                    if args.shading
                        plotShadedArea( axis, ...
                                        tSpanPlot, ...
                                        XCsmth( :, k, c ), ...
                                        XCsmth( :, end, c ), ...
                                        compColours(s,:), ...
                                        name = names(s) );
                    end
                    width = 1.0;
                
                else
                    width = 0.5;
                end

                % plot gradation curve

                if any(strcmp( args.type, {'Predicted','Both'} ))
                    % plot predicted values
                    plot( axis, ...
                          fda.tSpanAdaptive, XC( :, k, c ), ...
                          Color = gray, ...
                          LineWidth = 0.5 );
                end
                if any(strcmp( args.type, {'Smoothed','Both'} ))
                    % plot smoothed curves
                    if isOuterCurve
                        % include in legend
                        l = l+1;
                        pltObj(l) = plot( axis, ...
                                          tSpanPlot, XCsmth( :, k, c ), ...
                                          Color = compColours(s,:), ...
                                          LineWidth = width, ...
                                          DisplayName = names(s));
                    else
                        % don't record it for the legend
                        plot( axis, ...
                                          tSpanPlot, XCsmth( :, k, c ), ...
                                          Color = compColours(s,:), ...
                                          LineWidth = width );
                    end

                end

            end

            % plot the mean curve
            l = l+1;
            pltObj(l) = plot( axis, ...
                              tSpanPlot, XCsmth( :, end, c ), ...
                              Color = black, ...
                              LineWidth = 1.0, ...
                              DisplayName = 'Mean' );

            hold( axis, 'off' );

            % finalise the plot with formatting, etc
            if args.legend && c==1 && i==1
                legend( axis, pltObj, Location = 'best' );
            end
            
            if c==1
                title( axis, ['Component ' num2str(i)] );
            end
            if c==thisModel.XChannels
                xlabel( axis, thisModel.Info.TimeLabel );
            end
            if i==1
                ylabel( axis, thisModel.Info.ChannelLabels(c) );
            end
            xlim( axis, [tSpanPlot(1) tSpanPlot(end)] );

            if ~isempty( thisModel.Info.ChannelLimits )
                ylim( axis, thisModel.Info.ChannelLimits(c,:) );
            end               

            axis.YAxis.TickLabelFormat = '%.1f';

            finalisePlot( axis );

        end

    end

end
