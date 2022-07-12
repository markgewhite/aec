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
    end


    methods

        function self = Investigation( name, path, parameters, searchValues, setup )
            % Construct an investigation comprised of evaluations
            arguments
                name            string
                path            string
                parameters      string
                searchValues
                setup           struct

            end

            % create a folder for this investigation
            path = fullfile( path, name );
            if ~isfolder( path )
                mkdir( path )
            end

            % initialize properties
            self.Name = name;
            self.Path = path;

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
            
            self.TrainingResults.ReconLoss = zeros( allocation );
            self.TrainingResults.ReconLossSmoothed = zeros( allocation );
            self.TrainingResults.ReconLossRegular = zeros( allocation );
            self.TrainingResults.AuxModelLoss = zeros( allocation );
            self.TrainingResults.ZCorrelation = zeros( allocation );
            self.TrainingResults.XCCorrelation = zeros( allocation );
            self.TrainingResults.ZCovariance = zeros( allocation );
            self.TrainingResults.XCCovariance = zeros( allocation );
            
            self.TestingResults = self.TrainingResults;

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
                thisEvaluation = self.Evaluations{ idxC{:} };
                
                self.TrainingResults.ReconLoss( idxC{:} ) = ...
                    thisEvaluation.TrainingEvaluation.ReconLoss;
                self.TestingResults.ReconLoss( idxC{:} ) = ...
                    thisEvaluation.TestingEvaluation.ReconLoss;

                self.TrainingResults.ReconLossSmoothed( idxC{:} ) = ...
                    thisEvaluation.TrainingEvaluation.ReconLossSmoothed;
                self.TestingResults.ReconLossSmoothed( idxC{:} ) = ...
                    thisEvaluation.TestingEvaluation.ReconLossSmoothed;

                self.TrainingResults.ReconLossRegular( idxC{:} ) = ...
                    thisEvaluation.TrainingEvaluation.ReconLossRegular;
                self.TestingResults.ReconLossRegular( idxC{:} ) = ...
                    thisEvaluation.TestingEvaluation.ReconLossRegular;
                
                self.TrainingResults.AuxModelLoss( idxC{:} ) = ...
                    thisEvaluation.TrainingEvaluation.AuxModelLoss;
                self.TestingResults.AuxModelLoss( idxC{:} ) = ...
                    thisEvaluation.TestingEvaluation.AuxModelLoss;

                self.TrainingResults.ZCorrelation( idxC{:} ) = ...
                    thisEvaluation.TrainingCorrelations.ZCorrelation;
                self.TestingResults.ZCorrelation( idxC{:} ) = ...
                    thisEvaluation.TestingCorrelations.ZCorrelation;

                self.TrainingResults.XCCorrelation( idxC{:} ) = ...
                    thisEvaluation.TrainingCorrelations.XCCorrelation;
                self.TestingResults.XCCorrelation( idxC{:} ) = ...
                    thisEvaluation.TestingCorrelations.XCCorrelation;

                self.TrainingResults.ZCovariance( idxC{:} ) = ...
                    thisEvaluation.TrainingCorrelations.ZCovariance;
                self.TestingResults.ZCovariance( idxC{:} ) = ...
                    thisEvaluation.TestingCorrelations.ZCovariance;

                self.TrainingResults.XCCovariance( idxC{:} ) = ...
                    thisEvaluation.TrainingCorrelations.XCCovariance;
                self.TestingResults.XCCovariance( idxC{:} ) = ...
                    thisEvaluation.TestingCorrelations.XCCovariance;

    
                % save the evaluations
                thisEvaluation.save( path, name );

            end

            self.clearPredictions;

            self.save;
            

        end


        function self = clearPredictions( self )
            % Save memory by clearing model predictions
            arguments
                self        Investigation
            end

            nEval = prod( self.SearchDims );
            for i = 1:nEval
                idx = getIndices( i, self.SearchDims );
                idxC = num2cell( idx );

                self.Evaluations{ idxC{:} }.Model = ...
                    self.Evaluations{ idxC{:} }.Model.clearPredictions;

            end

        end
            


        function save( self )
            % Save the investigation to a specified path
            arguments
                self        Investigation
            end

            % define a small structure for saving
            report.BaselineSetup = self.BaselineSetup;
            report.GridSearch = self.GridSearch;
            report.TrainingResults = self.TrainingResults;
            report.TestingResults = self.TestingResults;
            
            name = strcat( self.Name, "-Investigation" );
            save( fullfile( self.Path, name ), 'report' );

        end  


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
