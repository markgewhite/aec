% ************************************************************************
% Class: plotter
%
% Class defining the plots from the models
%
% ************************************************************************

classdef plotter

    properties
        ax
    end

    methods

        function self = plotter( nCodes, nChannels, args )
            % Initialize the figures and axes
            arguments
                nCodes
                nChannels
                args.newFigures   logical = false;
            end

            if arg.newFigures
                figure;
            else
                figure(3);
            end
            self.ax.pred = subplot( 2,2,1 );
            self.ax.cls = subplot( 2,2,3 );
            
            if args.newFigures
                figure;
            else
                figure(4);
            end

            self.ax.distZTrn = subplot( 2,2,1 );
            self.ax.distZTst = subplot( 2,2,3 );
            
            self.ax.comp = gobjects( nCodes, nChannels );
            [ rows, cols ] = sqdim( nCodes );
            
            for j = 1:nChannels
                figure(4+j);
                for i = 1:nCodes
                    self.ax.comp(i,j) = subplot( rows, cols, i );
                end
            end

        end

        function plotZDist( self, Z, name, stdize )
            % Plot the Z (latent codes) distribution
            arguments
                self
                Z
                name
                stdize
            end

            if nargin<4
                stdize = false;
            end
            if stdize
                Z = (Z-mean(Z,2))./std(Z,[],2);
                xLbl = 'Std Z';
            else
                xLbl = 'Z';
            end
            
            nPts = 101;
            nCodes = size( Z, 1 );
            
            
            hold( ax, 'off');
            for i = 1:nCodes
                pdZ = fitdist( Z(i,:)', 'Kernel', 'Kernel', 'epanechnikov' );
                ZMin = prctile( Z(i,:), 0.0001 );
                ZMax = prctile( Z(i,:), 99.9999 );
                ZPts = ZMin : (ZMax-ZMin)/(nPts-1) : ZMax;
                Y = pdf( pdZ, ZPts );
                Y = Y/sum(Y);
                plot( ax, ZPts, Y, 'LineWidth', 1 );
                hold( ax, 'on' );
            end
            hold( ax, 'off');
            
            ylim( ax, [0 0.1] );
            
            title( ax, name );
            xlabel( ax, xLbl );
            ylabel( ax, 'Q(Z)' );
            
            
            drawnow;
            
        end

        
        function plotClusters( ax, Z, C, CHat, compact )

            if nargin>=4
                if isa( CHat, 'logical' )
                    inclPredictedClass = false;
                    compact = CHat;
                else
                    inclPredictedClass = true;
                    if nargin==4
                        compact = false;
                    end
                end
            else
                inclPredictedClass = false;
                compact = false;
            end
            
            C = categorical( C );
            class = sort( unique( C ) );
            
            cla( ax, 'reset' );
            hold( ax, 'on' );
            
            if compact
                dotSize = 10;
                dotSize2 = 4;
            else
                dotSize = 40;
                dotSize2 = 10;
            end
            
            % plot true classes (large dots)
            ax.ColorOrderIndex = 1;
            for i = 1:length( class )
                idx = (C==class(i));
                scatter( ax, Z(idx,1), Z(idx,2), dotSize, 'filled' );
            end
            
            if inclPredictedClass
                % plot estimated classes (small dots on top)
                CHat = categorical( CHat );
                ax.ColorOrderIndex = 1;
                for i = 1:length( class )
                    idx = (CHat==class(i));
                    scatter( ax, Z(idx,1), Z(idx,2), dotSize2, 'filled' );
                end
            end
            
            if ~compact
                % then add the centroids on top
                for i = 1:length( class )
                    idx = (C==class(i));
                    text( ax, mean( Z(idx,1) ), mean( Z(idx,2) ), ...
                              class(i), ...
                              'HorizontalAlignment', 'center', ...
                              'FontWeight', 'bold', ...
                              'FontSize', 10, ...
                              'Color', [0 0 0] );
                end
            end
            
            hold( ax, 'off' );
            
            if ~compact
                legend( ax, class, 'Location', 'Best' );
            end
            
            end


    end

end
