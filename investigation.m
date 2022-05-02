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
            self.TrainingResults.AuxModelLoss = zeros( allocation{:} );

            self.TestingResults = self.TrainingResults;
            

            for i = 1:nEval

                idx = getIndices( i, self.SearchDims );

                % apply the respective settings
                for j = 1:self.NumParameters

                    setup = applySetting( setup, ...
                                          parameters{j}, ...
                                          searchValues{j}(idx(j)) );
                
                end

                % carry out the evaluation
                idxC = num2cell( idx );
                self.Evaluations( idxC{:} ) = ...
                        run( self.Evaluations( idxC{:} ), setup );

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

                self.TrainingResults.AuxModelLoss( idxC{:} ) = ...
                    thisEvaluation.TrainingEvaluation.AuxModelLoss;
                self.TestingResults.AuxModelLoss( idxC{:} ) = ...
                    thisEvaluation.TestingEvaluation.AuxModelLoss;
                

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
        switch class( value )
            case {'double', 'string'}
                setup.(var).(remainder) = value;
            case 'cell'
                setup.(var).(remainder) = value{1};
        end
    end

end
