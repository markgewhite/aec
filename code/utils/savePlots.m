function savePlots( gObjs, path, name )
    % Save plots to a specified path
    % Plots are specified in a structure of graphics objects
    % with expected field names
    arguments
        gObjs       struct
        path        string {mustBeFolder}
        name        string
    end

    fullname = strcat( name, '.pdf' );

    if isfield( gObjs, 'ZDistribution' )
        % save the Z distribution plot
        fullpath = strcat( path, '/zdist/' );
        if ~isfolder( fullpath )
            mkdir( fullpath)
        end
        exportgraphics( gObjs.ZDistribution, ...
                        fullfile( fullpath, fullname ), ...
                        ContentType= 'vector', ...
                        Resolution = 300 );
    end

    if isfield( gObjs, 'ZClustering' )
        % save the Z clustering plot
        fullpath = strcat( path, '/zclust/' );
        if ~isfolder( fullpath )
            mkdir( fullpath)
        end
        exportgraphics( gObjs.ZClustering, ...
                        fullfile( fullpath, fullname ), ...
                        ContentType= 'vector', ...
                        Resolution = 300 );
    end


    if isfield( gObjs, 'LossFig' )

        % save the loss plots
        fullpath = strcat( path, '/loss/' );
        if ~isfolder( fullpath )
            mkdir( fullpath)
        end
        exportgraphics( gObjs.LossFig, ...
                    fullfile( fullpath, fullname ), ...
                    ContentType= 'vector', ...
                    Resolution = 300 );
        
    end

    % save the components (as a whole and individually)
    fullpath = strcat( path, '/comp/' );
    if ~isfolder( fullpath )
        mkdir( fullpath)
    end
    if isfield( gObjs, 'Components' )

        exportgraphics( gObjs.Components, ...
                        fullfile( fullpath, fullname ), ...
                        ContentType= 'vector', ...
                        Resolution = 300 );

    end

    if isfield( gObjs, 'Comp' )
        % individual components
        for i = 1:length(gObjs.Comp)
    
            fullname = strcat( name, '-C', num2str(i,'%02d'), '.pdf' );
            exportgraphics( gObjs.Comp(i), ...
                        fullfile( fullpath, fullname ), ...
                        ContentType= 'vector', ...
                        Resolution = 300 );
    
    
        end
    end

end   
