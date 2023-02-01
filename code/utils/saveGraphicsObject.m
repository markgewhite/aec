function saveGraphicsObject( obj, path, name )
    % Save plot as a PDF and a Matlab figure

    if ~isfolder( path )
        mkdir( path )
    end

    filename = strcat( fullfile( path, name ), '.pdf' );
    warning( 'off', 'MATLAB:print:ExportExcludesUI' );
    exportgraphics( obj, ...
                    filename, ...
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

    figPath = fullfile( path, 'figs' );
    if ~isfolder( figPath )
        mkdir( figPath)
    end

    filename = strcat( fullfile( figPath, name ), '.fig' );
    savefig( fig, filename );

end