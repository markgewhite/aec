function savePlots( gObjs, path, name )
    % Save plots to a specified path
    % Plots are specified in a structure of graphics objects
    % with expected field names
    arguments
        gObjs       struct
        path        string {mustBeFolder}
        name        string
    end

    if isfield( gObjs, 'ZDistribution' )
        % save the Z distribution plot
        fullpath = strcat( path, '/zdist/' );
        if ~isfolder( fullpath )
            mkdir( fullpath)
        end
        saveGraphicsObject( gObjs.ZDistribution, ...
                            fullfile( fullpath, name ) );
    end

    if isfield( gObjs, 'ZClustering' )
        % save the Z clustering plot
        fullpath = strcat( path, '/zclust/' );
        if ~isfolder( fullpath )
            mkdir( fullpath)
        end
        saveGraphicsObject( gObjs.ZClustering, ...
                            fullfile( fullpath, name ) );
    end


    if isfield( gObjs, 'LossFig' )

        % save the loss plots
        fullpath = strcat( path, '/loss/' );
        if ~isfolder( fullpath )
            mkdir( fullpath)
        end
        saveGraphicsObject( gObjs.LossFig, ...
                            fullfile( fullpath, name ) );
        
    end

    % save the components (as a whole and individually)
    fullpath = strcat( path, '/comp/' );
    if ~isfolder( fullpath )
        mkdir( fullpath)
    end
    if isfield( gObjs, 'Components' )

        saveGraphicsObject( gObjs.Components, ...
                            fullfile( fullpath, name ) );

    end

end   


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

    savefig( fig, strcat( filename, '.fig' ) );


end