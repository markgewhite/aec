function self = run( self )
    % Run a parallel grid search
    arguments
        self            ParallelInvestigation
    end

    % check if any evaluations have already been done
    isToBeDone = cellfun(@(x) ~isa(x, 'ModelEvaluation'), self.Evaluations );
    toDoIdx = find( isToBeDone );

    % use temporary flattened arrays for parallel procesing
    % so the indexing is unambiguous
    path = self.Path;
    setups = self.Setups(isToBeDone);
    names = self.EvaluationNames(isToBeDone);
    catchErrors = self.CatchErrors;
    memorySaving = self.MemorySaving;

    nEval = sum( isToBeDone, 'all' );
    thisEvaluation = cell( nEval, 1 );

    % run the evaluation loop
    parfor i = 1:nEval

        try
            argsCell = namedargs2cell( setups{i}.eval.args );
        catch
            argsCell = {};
        end

        disp(['Running evaluation = ' char(names(i)) ' ...']);
        if catchErrors
            try
                thisEvaluation{i} = ModelEvaluation( names(i), ...
                                                     path, ...
                                                     setups{i}, ...
                                                     argsCell{:} );
            catch ME
                warning(['***** Evaluation failed in ' char(names(i)) ' *****'])
                disp(['Error Message: ' ME.message]);
                for k = 1:length(ME.stack)
                    disp([ME.stack(k).name ', (line ' ...
                                         num2str(ME.stack(k).line) ')']);
                end
                continue
            end
        
        else
            thisEvaluation{i} = ModelEvaluation( names(i), ...
                                                 path, ...
                                                 setups{i}, ...
                                                 argsCell{:} );

        end

        % save the evaluations
        thisEvaluation{i}.conserveMemory( memorySaving );
        thisEvaluation{i}.save;

    end

    % store the results
    for i = 1:nEval

        idx = getIndices( toDoIdx(i), self.SearchDims );
        idxC = num2cell( idx );

        self.Evaluations{ idxC{:} } = thisEvaluation{i};
        try
            self.logResults( idxC, size(self.Setups) );
        catch
            disp(['Unable to log results for ' char(names(i))]);
        end

    end

    % save the current state of the investigation
    self.save( memorySaving = 1 );
    
end   