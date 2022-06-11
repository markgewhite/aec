classdef CompactRepresentationModel
    % Super class encompassing all individual dimensional reduction models

    properties
        XInputDim       % X dimension (number of points) for input
        XTargetDim      % X dimension for output
        ZDim            % Z dimension (number of features)
        CDim            % C dimension (number of classes)
        XChannels       % number of channels in X
        Scale           % scaling factor for reconstruction loss
        AuxModelType    % type of auxiliary model to use
        ShowPlots       % flag whether to show plots
        Figs            % figures holding the plots
        Axes            % axes for plotting latent space and components
        NumCompLines    % number of lines in the component plot

        LatentComponents % computed components across partitions
        TrainingLoss     % final training loss per partition
        ValidationLoss   % final validation loss per partition
    end

    methods

        function self = CompactRepresentationModel( theFullModel )
            % Initialize the model
            arguments
                theFullModel        FullRepresentationModel
            end

            self.XInputDim = theFullModel.XInputDim;
            self.XTargetDim = theFullModel.XTargetDim;
            self.ZDim = theFullModel.ZDim;
            self.CDim = theFullModel.CDim;
            self.XChannels = theFullModel.XChannels;
            self.Scale = theFullModel.Scale;
            self.AuxModelType = theFullModel.AuxModelType;
            self.ShowPlots = theFullModel.ShowPlots;
            self.Figs = theFullModel.Figs;
            self.Axes = theFullModel.Axes;
            self.NumCompLines = theFullModel.NumCompLines;

        end


        function err = getReconLoss( self, X, XHat )
            % Compute the  - placeholder
            arguments
                self        CompactRepresentationModel
                X           double
                XHat        double
            end

            err = mean( (XHat-X).^2, 'all' );
        
        end


        function [ varProp, compVar ] = getExplainedVariance( self, thisDataset )
            % Compute the explained variance for the components
            % using a finer-grained set of offsets
            arguments
                self            CompactRepresentationModel            
                thisDataset     modelDataset
            end

            % generate the latent encodings
            Z = self.encode( thisDataset );

            % generate the AE components
            [ XC, offsets ] = self.latentComponents( ...
                                            Z, ...
                                            sampling = 'Fixed', ...
                                            nSample = 100, ...
                                            centre = false, ...
                                            convert = true );

            % smooth the curves        
            XCFd = smooth_basis( thisDataset.TSpan.Target, ...
                                 XC, ...
                                 thisDataset.FDA.FdParamsTarget );
            XCReg = squeeze( ...
                        eval_fd( thisDataset.TSpan.Regular, XCFd ) );
            XReg = squeeze( ...
                        eval_fd( thisDataset.TSpan.Regular, ...
                                 thisDataset.XFd ) );

            % compute the components' explained variance
            [varProp, compVar] = self.explainedVariance( XReg, XCReg, offsets );    

        end


        function [ varProp, compVar ] = explainedVariance( self, X, XC, offsets )
            % Compute the explained variance for the components
            arguments
                self            CompactRepresentationModel
                X
                XC
                offsets         double
            end

            % convert to double for convenience
            if isa( X, 'dlarray' )
                X = double( extractdata( X ) );
            end  
            if isa( XC, 'dlarray' )
                XC = double( extractdata( XC ) );
            end  

            % re-order the dimensions for FDA
            if size( XC, 3 ) > 1
                X = permute( X, [1 3 2] );
                XC = permute( XC, [1 3 2] );
            end

            if mod( size( XC, 2 ), 2 )==1
                % remove the XC mean curve at the end
                if size( XC, 3 ) > 1
                    XC = XC( :, :, 1:end-1 );
                else
                    XC = XC( :, 1:end-1 );
                end
            end

            % centre using the X mean curve (XC mean is almost identical)
            X = X - mean( X, 2 );
            XC = XC - mean( XC, 2 );
            
            % compute the total variance from X
            totVar = mean( sum( X.^2 ) );

            % reshape XC by introducing dim for offset
            nOffsets = length( offsets );
            XC = reshape( XC, size(XC,1), self.XChannels, nOffsets, self.ZDim );

            % compute the component variances in turn
            compVar = zeros( self.XChannels, nOffsets, self.ZDim );

            for i = 1:self.XChannels
                for j = 1:nOffsets
                    for k = 1:self.ZDim
                        compVar( i, j, k ) = sum( (XC(:,i,j,k)/offsets(j)).^2 );
                    end
                end
            end

            compVar = squeeze( compVar./totVar );
            varProp = mean( compVar );

        end


        function plotZDist( self, Z, args )
            % Update the Z distributions plot
            arguments
                self                CompactRepresentationModel
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
                self                CompactRepresentationModel
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
            tSpanPlot = linspace( tSpan.Original(1), tSpan.Original(end), 101 );
            XCFd = smooth_basis( tSpan.Target, XC, fdParams );
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
                self                CompactRepresentationModel
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

        % Train the model on the data provided
        self = train( self, thisDataset )

        % Encode features Z from X using the model - placeholder
        Z = encode( self, X )

        % Reconstruct X from Z using the model - placeholder
        XHat = reconstruct( self, Z )

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