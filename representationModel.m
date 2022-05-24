classdef representationModel
    % Super class encompassing dimensional reduction models

    properties
        XDim            % X dimension (number of points)
        ZDim            % Z dimension (number of features)
        CDim            % C dimension (number of classes)
        XChannels       % number of channels in X
        ShowPlots       % flag whether to show plots
        Figs            % figures holding the plots
        Axes            % axes for plotting latent space and components
        NumCompLines    % number of lines in the component plot
    end

    methods

        function self = representationModel( args )
            % Initialize the model
            arguments
                args.XDim           double ...
                    {mustBeInteger, mustBePositive} = 10
                args.ZDim           double ...
                    {mustBeInteger, mustBePositive} = 1
                args.CDim           double ...
                    {mustBeInteger, mustBePositive} = 1
                args.XChannels      double ...
                    {mustBeInteger, mustBePositive} = 1
                args.NumCompLines   double...
                    {mustBeInteger, mustBePositive} = 8
                args.ShowPlots      logical = true
            end

            self.XDim = args.XDim;
            self.ZDim = args.ZDim;
            self.CDim = args.CDim;
            self.XChannels = args.XChannels;
            self.NumCompLines = args.NumCompLines;
            self.ShowPlots = args.ShowPlots;

            if args.ShowPlots
                [self.Figs, self.Axes] = ...
                        initializePlots( self.XChannels, self.ZDim );
            end

        end


        function err = getReconLoss( self, X, XHat )
            % Compute the  - placeholder
            arguments
                self        representationModel
                X           double
                XHat        double
            end

            err = mean( (XHat-X).^2, 'all' );
        
        end


        function plotZDist( self, Z, args )
            % Update the Z distributions plot
            arguments
                self                representationModel
                Z
                args.name           string = 'Latent Distribution'
                args.standardize    logical = false
                args.pdfLimit       double = 0.05
            end

            if isa( Z, 'dlarray' )
                Z = double( extractdata( Z ) )';
            end

            if args.standardize
                Z = (Z-mean(Z))./std(Z);
                xAxisLbl = 'Std Z';
            else
                xAxisLbl = 'Z';
            end
        
            nPts = 101;
            nCodes = size( Z, 2 );
             
            axis = self.Axes.ZDistribution;
            hold( axis, 'off');
            for i = 1:nCodes

                % fit a distribution
                pdZ = fitdist( Z(:,i), 'Kernel', 'Kernel', 'epanechnikov' );

                % get extremes
                Z01 = prctile( Z(:,i), 1 );
                Z50 = prctile( Z(:,i), 50 );
                Z99 = prctile( Z(:,i), 99 );

                % go well beyond range
                ZMin = Z50 - 2*(Z50-Z01);
                ZMax = Z50 + 2*(Z99-Z50);

                % evaluate the probability density function
                ZPts = linspace( ZMin, ZMax, nPts );
                Y = pdf( pdZ, ZPts );
                Y = Y/sum(Y);

                plot( axis, ZPts, Y, 'LineWidth', 1 );
                hold( axis, 'on' );

            end
            
            hold( axis, 'off');
            
            ylim( axis, [0 args.pdfLimit] );
            
            title( axis, args.name );
            xlabel( axis, xAxisLbl );
            ylabel( axis, 'Q(Z)' );   
            axis.YAxis.TickLabelFormat = '%.2f';
            
            finalisePlot( axis, square = true );
            
        end


        function plotLatentComp( self, XC, tSpan, fdParams, args )
            % Plot characteristic curves of the latent codings which are similar
            % in conception to the functional principal components
            arguments
                self                representationModel
                XC
                tSpan               struct
                fdParams            
                args.nSample        double = 0
                args.type           char ...
                    {mustBeMember(args.type, ...
                        {'Smoothed', 'Predicted', 'Both'} )} = 'Smoothed'
                args.shading        logical = false
                args.legend         logical = true
                args.plotTitle      string = "<Dataset>"
                args.xAxisLabel     string = "Time"
                args.yAxisLabel     string = "<Channel>"
                args.yAxisLimits    double
    
            end
            
            if isa( XC, 'dlarray' )
                XC = double( extractdata( XC ) );
            end
    
            % re-order the dimensions for FDA
            if size( XC, 3 ) > 1
                XC = permute( XC, [1 3 2] );
            end
    
            % smooth and re-evaluate all curves
            tSpanPlot = linspace( tSpan.original(1), tSpan.original(end), 101 );
            XCFd = smooth_basis( tSpan.target, XC, fdParams );
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
                nSample = self.NumCompLines;
            end

            for c = 1:self.XChannels

                k = 0; % sample counter

                for i = 1:self.ZDim
    
                    axis = self.Axes.Comp(c,i);
    
                    cla( axis );
                    hold( axis, 'on' );

                    pltObj = gobjects( 3, 1 );

                    l = 0; % legend counter
                    s = 1; % sign/colour counter

                    % plot the gradations
                    for j = 1:nSample
                        
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
                            width = 1.5;
                        
                        else
                            width = 1.0;
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
        
                        if mod( j, nSample/2 )==0
                            % next colour
                            s = s+1;
                        end

                    end
    
                    % plot the mean curve
                    l = l+1;
                    pltObj(l) = plot( axis, ...
                                      tSpanPlot, XCsmth( :, end, c ), ...
                                      Color = black, ...
                                      LineWidth = 1.5, ...
                                      DisplayName = 'Mean' );
        
                    hold( axis, 'off' );
        
                    % finalise the plot with formatting, etc
                    if args.legend && c==1 && i==1
                        legend( axis, pltObj, Location = 'best' );
                    end
                    
                    if c==1
                        title( axis, ['Component ' num2str(i)] );
                    end
                    if c==self.XChannels
                        xlabel( axis, args.xAxisLabel );
                    end
                    if i==1
                        ylabel( axis, args.yAxisLabel(c) );
                    end
                    xlim( axis, [tSpanPlot(1) tSpanPlot(end)] );
        
                    if ~isempty( args.yAxisLimits )
                        ylim( axis, args.yAxisLimits(c,:) );
                    end               
        
                    axis.YAxis.TickLabelFormat = '%.1f';
        
                    finalisePlot( axis );
    
                end

            end

        end
    

        function plotZClusters( self, Z, args )
            % Plot the latent codes on 2D plane
            arguments
                self                representationModel
                Z
                args.Y              = []
                args.type           char ...
                    {mustBeMember(args.type, ...
                        {'Canonical', 'TSNE'} )} = 'TSNE'
                args.name           string = 'Latent Space'
                args.perplexity     double = 50
                args.compact        logical = false

            end

            if isa( Z, 'dlarray' )
                Z = double( extractdata( Z ) )';
            end
            % ensure Z is 2D
            Z = reshape( Z, size(Z,1), [] );

            if isempty( args.Y ) && strcmp( args.type, 'Canonical' )
                eid = 'aeModel:LabelsMissing';
                msg = 'Canonical discriminant analysis needs labels.';
                throwAsCaller( MException(eid,msg) );
            end

            if ~isempty( args.Y )
                Y = args.Y;
                if isa( args.Y, 'dlarray' )
                    Y = double( extractdata( Y ) );
                end
                classes = unique( Y );
            else
                Y = ones( size( Z,1 ), 1 );
                classes = 1;
            end           

            switch args.type
                case 'Canonical'
                    % canonical discriminant analysis                  
                    ZCanInfo = cda( Z, Y );
                    ZT = ZCanInfo.scores;
                    if size( ZT, 2 )==1
                        % only one canonical dimension
                        ZT = [ ZT ZT ];  
                    end

                case 'TSNE'
                    % t-distribution stochastic neighbour embedding
                    perplexity = min( size(Z,1), args.perplexity );
                    ZT = tsne( Z, ...
                               Perplexity = perplexity, ...
                               Standardize = true ); 
                    
            end

            axis = self.Axes.ZClustering;
            cla( axis, 'reset' );
            hold( axis, 'on' );
            
            if args.compact
                dotSize = 5;
            else
                dotSize = 10;
            end
            
            % plot true classes (large dots)
            colours = lines( length(classes) );
            gscatter( axis, ZT(:,1), ZT(:,2), Y, colours, '.', dotSize );
            
            hold( axis, 'off' );
            
            if ~args.compact
                legend( axis, 'Location', 'Best' );
            end

            title( axis, args.name );
            axis.TickDir = 'none';
            axis.XTickLabel = [];
            axis.YTickLabel = [];
            finalisePlot( axis, square = true );


        end

    end


    methods (Abstract)

        % Train the model - placeholder    
        self = train( self, X )

        % Encode features Z from X using the model - placeholder
        Z = encode( self, X )

        % Reconstruct X from Z using the model - placeholder
        XHat = reconstruct( self, Z )

    end

end


function [figs, axes]= initializePlots( XChannels, ZDim )
    % Setup plots for latent space and components
   
    % setup figure for Z distribution and clustering
    figs.LatentSpace = figure(1);
    clf;
    axes.ZDistribution = subplot( 1, 2, 1 );
    axes.ZClustering = subplot( 1, 2, 2 );

    % setup the components figure
    figs.Components = figure(2);
    figs.Components.Position(2) = 0;
    figs.Components.Position(3) = 100 + ZDim*250;
    figs.Components.Position(4) = 100 + XChannels*250;
    
    clf;
    axes.Comp = gobjects( XChannels, ZDim );

    for i = 1:XChannels
        for j = 1:ZDim
            axes.Comp(i,j) = subplot( XChannels, ZDim, (i-1)*ZDim + j );
        end
    end

end


function obj = plotShadedArea( ax, t, y1, y2, colour, args )
    % Plot a shaded area
    arguments
        ax          
        t               double
        y1              double
        y2              double
        colour          double  
        args.alpha      double = 0.25
        args.name       char
    end

    % set the boundary
    tRev = [ t, fliplr(t) ];
    yRev = [ y1; flipud(y2) ];
        
    % draw shaded region
    obj = fill( ax, tRev, yRev, colour, ...
                    'FaceAlpha', args.alpha, ...
                    'EdgeColor', 'none', ...
                    'DisplayName', args.name );

end


function finalisePlot( ax, args )
    arguments
        ax
        args.square     logical = false
    end

    ax.Box = false;
    ax.TickDir = 'out';
    ax.XAxis.LineWidth = 1;
    ax.YAxis.LineWidth = 1;
    ax.FontName = 'Arial';
    if args.square
        ax.PlotBoxAspectRatio = [1 1 1];
    end

end