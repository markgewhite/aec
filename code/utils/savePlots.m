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
        saveGraphicsObject( gObjs.ZDistribution, ...
                            fullpath, name );
    end

    if isfield( gObjs, 'ZClustering' )
        % save the Z clustering plot
        fullpath = strcat( path, '/zclust/' );
        saveGraphicsObject( gObjs.ZClustering, ...
                            fullpath, name );
    end

    if isfield( gObjs, 'LossFig' )
        % save the loss plots
        fullpath = strcat( path, '/loss/' );
        saveGraphicsObject( gObjs.LossFig, ...
                            fullpath, name );       
    end

    if isfield( gObjs, 'Components' )
        % save the components (as a whole and individually)
        fullpath = strcat( path, '/comp/' );
        saveGraphicsObject( gObjs.Components, ...
                            fullpath, name );

    end

end   
