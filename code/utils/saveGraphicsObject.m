function saveGraphicsObject( obj, filename )

    exportgraphics( obj, ...
                    strcat( filename, '.pdf' ), ...
                    ContentType= 'vector', ...
                    Resolution = 300 );

    if isa( obj, 'matlab.graphics.axis.Axes' )
        fig = obj.Parent;
    else
        fig = obj;
    end

    if isa( fig, 'matlab.graphics.layout.TiledChartLayout' )
        fig = fig.Parent;
    end

    savefig( fig, strcat( filename, '.fig' ) );


end