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
        TrainingResults     % structure summarising results from evaluations
        TestingResults      % structure summarising results from evaluations
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

            if length( nDimsCell ) > 1
                allocation = nDimsCell;
            else
                allocation = [ nDimsCell,1 ];
            end
            self.TrainingResults.LossTrace = cell( allocation{:} );
            self.TrainingResults.ReconLoss = zeros( allocation{:} );
            self.TrainingResults.ReconLossSmoothed = zeros( allocation{:} );
            self.TrainingResults.AuxModelLoss = zeros( allocation{:} );

            self.TestingResults = self.TrainingResults;
            

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

                % carry out the evaluation
                idxC = num2cell( idx );
                self.Evaluations( idxC{:} ) = ...
                        run( self.Evaluations( idxC{:} ), setup, true );

                % record results
                thisEvaluation = self.Evaluations( idxC{:} );

                if isfield( thisEvaluation.Model, 'trainer' )
                    self.TrainingResults.LossTrace{ idxC{:} } = ...
                        thisEvaluation.Model.trainer.lossTrn;
                    self.TestingResults.LossTrace{ idxC{:} } = ...
                        thisEvaluation.Model.trainer.lossVal;
                end
                
                self.TrainingResults.ReconLoss( idxC{:} ) = ...
                    thisEvaluation.TrainingEvaluation.ReconLoss;
                self.TestingResults.ReconLoss( idxC{:} ) = ...
                    thisEvaluation.TestingEvaluation.ReconLoss;

                self.TrainingResults.ReconLossSmoothed( idxC{:} ) = ...
                    thisEvaluation.TrainingEvaluation.ReconLossSmoothed;
                self.TestingResults.ReconLossSmoothed( idxC{:} ) = ...
                    thisEvaluation.TestingEvaluation.ReconLossSmoothed;

                self.TrainingResults.AuxModelLoss( idxC{:} ) = ...
                    thisEvaluation.TrainingEvaluation.AuxModelLoss;
                self.TestingResults.AuxModelLoss( idxC{:} ) = ...
                    thisEvaluation.TestingEvaluation.AuxModelLoss;

    
                % save the plots
                % in a folder specific to the evaluation
                folder = strcat( name, '(' );
                for j = 1:length(idx)
                    folder = strcat( folder, num2str(idx(j)) );
                    if j < length(idx)
                        folder = strcat( folder, ',' );
                    end
                end
                folder = strcat( folder, ')');

                fullpath = fullfile(path, folder);

                if ~isfolder( fullpath )
                    mkdir( fullpath )
                end

                thisEvaluation.save( fullpath, name );


            end
            


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


function setup = updateDependencies( setup, parameter, value )
    % Apply additional settings due to dependencies
    % Assess them programmatically
    arguments
        setup       struct
        parameter   string
        value       
    end

    switch parameter

        case 'model.class'

            switch func2str( value{1} )

                case {'fcModel', 'convModel'}

                    dependency = 'data.args.hasNormalizedInput';
                    reqValue = true;
                    setup = applySetting( setup, dependency, reqValue );

                case 'lstmModel'

                    dependency = 'trainer.args.partialBatch';
                    reqValue = 'discard';
                    setup = applySetting( setup, dependency, reqValue );

                case 'pcaModel'

                    dependency = 'data.args.hasNormalizedInput';
                    reqValue = true;
                    setup = applySetting( setup, dependency, reqValue );
                    
                    dependency = 'data.args.hasMatchingOutput';
                    reqValue = true;
                    setup = applySetting( setup, dependency, reqValue );

            end

    end

end
