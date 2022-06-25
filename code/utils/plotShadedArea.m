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
    obj = patch( ax, tRev, yRev, colour, ...
                    'FaceAlpha', args.alpha, ...
                    'EdgeColor', 'none', ...
                    'DisplayName', args.name );

end