classdef Investigation
    % Class defining a model investigation, a grid search

    properties
        Name                % name of the investigation
        Path                % file path for storing results
        NumParameters       % number of search parameters
        SearchDims          % dimensions of the parameter search
        Parameters          % named parameters
        GridSearch          % the grid search values for those parameters
        BaselineSetup       % structure recording the baseline
        Setups              % structure of all individual evaluations
        NumEvaluations      % total number of grid searches
        EvaluationNames     % names of all evaluations
        Evaluations         % array of evaluation objects
        TrainingResults     % structure summarising results from evaluations
        TestingResults      % structure summarising results from evaluations
        CatchErrors         % flag indicating if try-catch should be used
        MemorySaving        % memory saving level
    end


    methods

        function self = Investigation( name, path, parameters, ...
                                       searchValues, setup, ...
                                       catchErrors, memorySaving )
            % Construct an investigation comprised of evaluations
            arguments
                name            string
                path            string
                parameters      string
                searchValues
                setup           struct
                catchErrors     logical = false
                memorySaving    double {mustBeInteger, ...
                                mustBeInRange( memorySaving, 0, 3 )} = 0
            end

            % initialize properties
            self.Name = name;
            self.Path = path;
            self.CatchErrors = catchErrors;
            self.MemorySaving = memorySaving;

            % create a folder for this investigation
            setup.model.args.path = fullfile( path, name );
            if ~isfolder( setup.model.args.path )
                mkdir( setup.model.args.path )
            end

            self.BaselineSetup = setup;
            self.Parameters = parameters;
            self.GridSearch = searchValues;
            self.TrainingResults.Mean = [];
            self.TrainingResults.SD = [];
            self.TestingResults.Mean = [];
            self.TestingResults.SD = [];

            % setup the grid search
            self.NumParameters = length( parameters );
            self.SearchDims = cellfun( @length, self.GridSearch ); 
            self.NumEvaluations = prod( self.SearchDims );

            % initialize evaluation arrays 
            if length( self.SearchDims ) > 1
                allocation = self.SearchDims;
            else
                allocation = [ self.SearchDims, 1 ];
            end

            self.Evaluations = cell( allocation );
            self.EvaluationNames = strings( allocation );
            self.Setups = cell( allocation );

            for i = 1:self.NumEvaluations
        
                idx = getIndices( i, self.SearchDims );
                idxC = num2cell( idx );

                self.EvaluationNames( idxC{:}) = strcat( self.Name, constructName(idx) );

                % apply the respective settings
                self.Setups{ idxC{:} } = self.BaselineSetup;
                for j = 1:self.NumParameters
        
                    self.Setups{ idxC{:} } = applySetting( ...
                                                    self.Setups{ idxC{:} }, ...
                                                    self.Parameters{j}, ...
                                                    self.GridSearch{j}(idx(j)) );
        
                    self.Setups{ idxC{:} } = updateDependencies( ...
                                                    self.Setups{ idxC{:} }, ...
                                                    self.Parameters{j}, ...
                                                    self.GridSearch{j}(idx(j)) );
                
                end               
           
            end
            
        end           


        % class methods

        self = conserveMemory( self, level )

        datasets = getDatasets( self, args )
        
        report = getResults( self )

        model = linearModel( self, outcome, args )

        model = mixedModel( self, group, outcome, args )

        self = run( self )

        save( self )

        fig = saveDataPlot( self, args )

        report = saveReport( self )

    end

end