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
                                       searchValues, setup, ...
                                       memorySaving, resume )
            % Construct an investigation comprised of evaluations
            arguments
                name            string
                path            string
                parameters      string
                searchValues
                setup           struct
                memorySaving    double {mustBeInteger, ...
                    mustBeInRange( memorySaving, 0, 4 )} = 0
                resume          logical = false
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

            if resume
                self = self.load;
            else  
                self.BaselineSetup = setup;
                self.Parameters = parameters;
                self.GridSearch = searchValues;
                self.TrainingResults.Mean = [];
                self.TrainingResults.SD = [];
                self.TestingResults.Mean = [];
                self.TestingResults.SD = [];
            end

            % setup the grid search
            self.NumParameters = length( parameters );
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

                idx = getIndices( i, self.SearchDims );
                idxC = num2cell( idx );

                if ~isempty(self.TrainingResults.Mean)
                    if self.TrainingResults.Mean.ReconLoss(idxC{:})~=0
                        % evaluation already performed, move to next one
                        disp(['Evaluation (' num2str(idx') ...
                              ') previously completed.']);
                        continue
                    end
                end

                % apply the respective settings
                setup = self.BaselineSetup;
                for j = 1:self.NumParameters

                    setup = applySetting( setup, ...
                                          parameters{j}, ...
                                          searchValues{j}(idx(j)) );

                    setup = updateDependencies( setup, ...
                                               parameters{j}, ...
                                               searchValues{j}(idx(j)) );
                
                end

                % assign a folder for this evaluation
                %setup.model.args.path = folder( path, name, idx );

                % carry out the evaluation
                setup.model.args.name = strcat( name, constructName(idx) );
                try
                    argsCell = namedargs2cell( setup.eval.args );
                catch
                    argsCell = {};
                end

                %try
                    self.Evaluations{ idxC{:} } = ...
                                ModelEvaluation( setup.model.args.name, ...
                                                 setup, ...
                                                 argsCell{:} );
                    % record results               
                    self.TrainingResults.Mean = updateResults( ...
                            self.TrainingResults.Mean, idxC, allocation, ...
                            self.Evaluations{ idxC{:} }.CVLoss.Training.Mean );
                    self.TrainingResults.SD = updateResults( ...
                            self.TrainingResults.SD, idxC, allocation, ...
                            self.Evaluations{ idxC{:} }.CVLoss.Training.SD );
    
                    self.TrainingResults.Mean = updateResults( ...
                            self.TrainingResults.Mean, idxC, allocation, ...
                            self.Evaluations{ idxC{:} }.CVCorrelations.Training.Mean );
                    self.TrainingResults.SD = updateResults( ...
                            self.TrainingResults.SD, idxC, allocation, ...
                            self.Evaluations{ idxC{:} }.CVCorrelations.Training.SD );
    
                    self.TestingResults.Mean = updateResults( ...
                            self.TestingResults.Mean, idxC, allocation, ...
                            self.Evaluations{ idxC{:} }.CVLoss.Validation.Mean );
                    self.TestingResults.SD = updateResults( ...
                            self.TestingResults.SD, idxC, allocation, ...
                            self.Evaluations{ idxC{:} }.CVLoss.Validation.SD );
    
                    self.TestingResults.Mean = updateResults( ...
                            self.TestingResults.Mean, idxC, allocation, ...
                            self.Evaluations{ idxC{:} }.CVCorrelations.Validation.Mean );
                    self.TestingResults.SD = updateResults( ...
                            self.TestingResults.SD, idxC, allocation, ...
                            self.Evaluations{ idxC{:} }.CVCorrelations.Validation.SD );
        
                    % save the evaluations
                    self.Evaluations{ idxC{:} }.save( setup.model.args.path, name );
                
                %catch
                
                    %warning('Evaluation failed.')
                
                %end

                % conserve memory - essential in a long run
                if self.MemoryConservation == 4
                    % maximum conservation: erase the evaluation
                    % useful if the grid search is extensive
                    self.Evaluations{ idxC{:} } = [];
                else
                    % scaled memory conservation
                    self.Evaluations{ idxC{:} } = ...
                        self.Evaluations{ idxC{:} }.conserveMemory( ...
                                                self.MemoryConservation );
                end

                self.save;

            end
            
        end           


        % class methods

        datasets = getDatasets( self, args )
        
        report = getResults( self )


    end

end