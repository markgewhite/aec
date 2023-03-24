function self = run( self )
    % Run a parallel grid search
    arguments
        self            ParallelInvestigation
    end

    nEval = self.NumEvaluations;
    
    % use temporary flattened arrays for parallel procesing
    % so the indexing is unambiguous
    path = self.Path;
    setups = self.Setups(:);
    names = self.EvaluationNames(:);
    catchErrors = self.CatchErrors;
    memorySaving = self.MemorySaving;

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
                warning('*****!!!!! Evaluation failed !!!!!*****')
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

        disp(['Saving evaluation = ' char(names(i)) ' ...']);

        % save the evaluations
        thisEvaluation{i} = thisEvaluation{i}.conserveMemory( memorySaving );
        thisEvaluation{i}.save;

    end

    % store the results
    for i = 1:nEval

        idx = getIndices( i, self.SearchDims );
        idxC = num2cell( idx );

        self.Evaluations{ idxC{:} } = thisEvaluation{i};
        self = self.logResults( idxC, size(self.Setups) );

    end

    % save the current state of the investigation
    self.save( memorySaving = 1 );
    
end   