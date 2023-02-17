function plotZDist( self, Z, args )
    % Update the Z distributions plot
    arguments
        self                RepresentationModel
        Z                   {mustBeA( Z, { 'dlarray', 'double' })}
        args.name           string = 'Latent Distribution'
        args.standardize    logical = false
        args.pdfLimit       double = 0.05
    end

    if isa( Z, 'dlarray' )
        Z = double( extractdata(gather(Z)) )';
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

