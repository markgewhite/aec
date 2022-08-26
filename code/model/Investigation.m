classdef Investigation
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
        TrainingResults     % structure summarising results from evaluations
        TestingResults      % structure summarising results from evaluations
        MemoryConservation  % degree of memory conservation employed
                            % 0 = none; 1 = graphics cleared; 
                            % 2 = graphics and predictions cleared;
                            % 3 = graphics, predictions, and optimzer cleared
    end


    methods

        function self = Investigation( name, path, parameters, ...
                                       searchValues, setup, memorySaving )
            % Construct an investigation comprised of evaluations
            arguments
                name            string
                path            string
                parameters      string
                searchValues
                setup           struct
                memorySaving    double {mustBeInteger, ...
                    mustBeInRange( memorySaving, 0, 4 )} = 0
            end

            % create a folder for this investigation
            path = fullfile( path, name );
            if ~isfolder( path )
                mkdir( path )
            end

            % initialize properties
            self.Name = name;
            self.Path = path;
            self.MemoryConservation = memorySaving;

            % add the name and path to the model properties
            setup.model.args.name = name;
            setup.model.args.path = path;

            self.BaselineSetup = setup;

            self.Parameters = parameters;
            self.NumParameters = length( parameters );

            % setup the grid search
            self.GridSearch = searchValues;
            self.SearchDims = cellfun( @length, self.GridSearch ); 

            % initialize evaluation arrays 
            if length( self.SearchDims ) > 1
                allocation = self.SearchDims;
            else
                allocation = [ self.SearchDims, 1 ];
            end

            self.Evaluations = cell( allocation );

            nEval = prod( self.SearchDims );
            % run the evaluation loop
            for i = 1:nEval

                setup = self.BaselineSetup;
                idx = getIndices( i, self.SearchDims );

                % apply the respective settings
                for j = 1:self.NumParameters

                    setup = applySetting( setup, ...
                                          parameters{j}, ...
                                          searchValues{j}(idx(j)) );

                    setup = updateDependencies( setup, ...
                                               parameters{j}, ...
                                               searchValues{j}(idx(j)) );
                
                end

                % assign a folder for this evaluation
                setup.model.args.path = folder( path, name, idx );

                % carry out the evaluation
                idxC = num2cell( idx );
                evalName = strcat( name, constructName(idx) );
                self.Evaluations{ idxC{:} } = ...
                                ModelEvaluation( evalName, setup, true );

                % record results               
                self.TrainingResults = updateResults( ...
                        self.TrainingResults, idxC, allocation, ...
                        self.Evaluations{ idxC{:} }.TrainingEvaluation );
                self.TrainingResults = updateResults( ...
                        self.TrainingResults, idxC, allocation, ...
                        self.Evaluations{ idxC{:} }.TrainingCorrelations );

                self.TestingResults = updateResults( ...
                        self.TestingResults, idxC, allocation, ...
                        self.Evaluations{ idxC{:} }.TestingEvaluation );
                self.TestingResults = updateResults( ...
                        self.TestingResults, idxC, allocation, ...
                        self.Evaluations{ idxC{:} }.TestingCorrelations );
    
                % save the evaluations
                self.Evaluations{ idxC{:} }.save( setup.model.args.path, name );

                % conserve memory - essential in a long run
                if self.MemoryConservation == 4
                    % maximum conservation: erase the evaluation
                    % useful if the grid search is extensive
                    self.Evaluations{ idxC{:} } = [];
                else
                    % scaled memory conservation
                    self.Evaluations{ idxC{:} }.Model = ...
                        self.Evaluations{ idxC{:} }.Model.conserveMemory( ...
                                            self.MemoryConservation );
                end

            end

            self.save;
            

        end           


        function report = save( self )
            % Save the investigation to a specified path
            arguments
                self        Investigation
            end

            report = self.getResults;
            
            name = strcat( self.Name, "-Investigation" );
            save( fullfile( self.Path, name ), 'report' );

        end


        function report = getResults( self )
            % Return a structure summarizing the results
            arguments
                self        Investigation
            end

            % define a small structure for saving
            report.BaselineSetup = self.BaselineSetup;
            report.GridSearch = self.GridSearch;
            report.TrainingResults = self.TrainingResults;
            report.TestingResults = self.TestingResults;

        end


        function datasets = getDatasets( self, args )
            % Get the datasets used across all evaluations
            arguments
                self            Investigation
                args.which      string {mustBeMember( ...
                    args.which, {'First', 'All'} )} = 'First'
                args.set        string {mustBeMember( ...
                    args.set, {'Training', 'Testing'} )} = 'Testing'
            end

            fld = strcat( args.set, "Dataset" );

            switch args.which
                case 'First'
                    datasets = self.Evaluations{1}.(fld);
                                
                case 'All'
                    datasets = cell( self.SearchDims );
                    for i = 1:prod( self.SearchDims )
                        idx = getIndices( i, self.SearchDims );
                        datasets( idx{:} ) = self.Evaluations( idx{:} ).(fld);
                    end
            end

        end


    end


end


function results = updateResults( results, idx, allocation, info )
    % Update the ongoing results with the latest evaluation
    arguments
        results     struct
        idx         cell
        allocation  double
        info        struct
    end

    fld = fieldnames( info );
    isNew = isempty( results );
    if isNew
        results(1).temp = 0;
    end
    for i = 1:length(fld)
        if ~isfield( results, fld{i} )
            if length(info.(fld{i}))==1
                results.(fld{i}) = zeros( allocation );
            else
                results.(fld{i}) = cell( allocation );
            end
        end
        if ~iscell( results.(fld{i}) )
            results.(fld{i})(idx{:}) = info.(fld{i});
        else
            results.(fld{i}){idx{:}} = info.(fld{i});
        end
    end
    if isNew
        results = rmfield( results, 'temp' );
    end

end


function idx = getIndices( i, dims )
    % Convert counter to set of indices based on dimensions
    arguments
        i       double
        dims    double
    end

    if i>prod(dims)
        error('Requested index exceeds dimensions.');
    end

    nDim = length( dims );
    idx = zeros( nDim, 1 );
    for k = 1:nDim-1
        base = prod( dims(k+1:end) );
        idx(k) = ceil( i/base );
        i = mod( i, base );
        if idx(k)==0
            idx(k) = dims(k);
        end
    end
    if i==0
        idx(nDim) = dims(nDim);
    else
        idx(nDim) = i;
    end

end


function fullpath = folder( path, name, idx )
    % Create folder specific for the evaluation in question

    folder = strcat( name, '-Eval', constructName(idx) );
    fullpath = fullfile(path, folder);
    if ~isfolder( fullpath )
        mkdir( fullpath )
    end

end


function name = constructName( idx )
    % Construct a name from the indices

    name = '(';
    for j = 1:length(idx)
        name = strcat( name, num2str(idx(j)) );
        if j < length(idx)
            name = strcat( name, ',' );
        end
    end
    name = strcat( name, ')');

end
