classdef investigation
    % Class defining a model evaluation

    properties
        Name                % name of the investigation
        Path                % file path for storing results
        NumParameters       % number of search parameters
        SearchDims          % dimensions of the parameter search
        Parameters
        GridSearch
        BaselineSetup       % structure recording the setup
        Evaluations         % array of evaluation objects
    end


    methods

        function self = investigation( name, path, parameters, searchValues, setup )
            % Construct an investigation comprised of evaluations
            arguments
                name            string
                path            string {mustBeFolder}
                parameters      string
                searchValues
                setup           struct

            end

            self.Name = name;
            self.Path = path;

            self.BaselineSetup = setup;

            self.Parameters = parameters;
            self.NumParameters = length( parameters );

            self.GridSearch = searchValues;
            self.SearchDims = cellfun( @length, self.GridSearch ); 
            
            nEval = prod( self.SearchDims );
            nDimsCell = num2cell( self.SearchDims );
            initObj( nDimsCell{:} ) = modelEvaluation;
            self.Evaluations = initObj;

            for i = 1:nEval

                idx = getIndices( i, self.SearchDims );

                % apply the respective settings
                for j = 1:self.NumParameters

                    setup = applySetting( setup, ...
                                          parameters{j}, ...
                                          searchValues{j}(idx(j)) );
                
                end

                % carry out the evaluation
                idxCell = num2cell( idx );
                self.Evaluations( idxCell{:} ) = ...
                        run( self.Evaluations( idxCell{:} ), setup );

            end
            

            % save the evaluation object
            % save the plots
            % in a folder specific to the evaluation

        end


    end


end


function idx = getIndices( i, dims )
    % Convert counter to set of indices based on dimensions
    arguments
        i       double
        dims    double
    end

    nDim = length( dims );
    idx = zeros( nDim, 1 );
    for k = 1:nDim-1
        base = prod( dims(k+1:end) );
        idx(k) = ceil( i/base );
        i = mod( i, base );
    end
    if i==0
        idx(nDim) = dims(nDim);
    else
        idx(nDim) = i;
    end

end


function setup = applySetting( setup, parameter, value )
    % Apply the parameter value by recursively moving through structure
    arguments
        setup       struct
        parameter   string
        value       
    end

    var = extractBefore( parameter, "." );
    remainder = extractAfter( parameter, "." );
    if contains( remainder, "." )
        setup.(var) = applySetting( setup.(var), remainder, value );
    else
        setup.(var).(remainder) = value;
    end

end
