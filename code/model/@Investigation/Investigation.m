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
        CatchErrors         % flag indicating if try-catch should be used
        MemorySaving        % memory saving level
    end


    methods

        function self = Investigation( name, path, parameters, ...
                                       searchValues, setup, ...
                                       resume, catchErrors, memorySaving )
            % Construct an investigation comprised of evaluations
            arguments
                name            string
                path            string
                parameters      string
                searchValues
                setup           struct
                resume          logical = false
                catchErrors     logical = true
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
            for c = 1:nEval

                idx = getIndices( c, self.SearchDims );
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

                % carry out the evaluation
                setup.model.args.name = strcat( name, constructName(idx) );
                try
                    argsCell = namedargs2cell( setup.eval.args );
                catch
                    argsCell = {};
                end

                if catchErrors
                    try
                        thisEvaluation = ModelEvaluation( ...
                                                    setup.model.args.name, ...
                                                    setup.model.args.path, ...
                                                    setup, ...
                                                    argsCell{:} );
                    catch ME
                        warning('*****!!!!! Evaluation failed !!!!!*****')
                        disp(['Error Message: ' ME.message]);
                        for i = 1:length(ME.stack)
                            disp([ME.stack(i).name ', (line ' ...
                                                 num2str(ME.stack(i).line) ')']);
                        end
                        continue
                    end
                
                else
                    thisEvaluation = ModelEvaluation( ...
                                                setup.model.args.name, ...
                                                setup.model.args.path, ...
                                                setup, ...
                                                argsCell{:} );

                end
   
                % save the evaluations
                thisEvaluation = thisEvaluation.conserveMemory( self.MemorySaving );
                thisEvaluation.save;

                % record results
                self.Evaluations{ idxC{:} } = thisEvaluation;
                self = self.logResults( idxC, allocation );

                % save the current state of the investigation
                self.save( memorySaving = 1 );

            end
            
        end           


        % class methods

        self = conserveMemory( self, level )

        datasets = getDatasets( self, args )
        
        report = getResults( self )

        save( self )

        fig = saveDataPlot( self, args )

        report = saveReport( self )

    end

end