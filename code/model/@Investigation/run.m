function self = run( self )
    % Run the grid search
    arguments
        self            Investigation
    end

    % check if any evaluations have already been done
    isToBeDone = cellfun(@(x) ~isa(x, 'ModelEvaluation'), self.Evaluations );
    toDoIdx = find( isToBeDone );

    % run the evaluation loop
    nEval = sum( isToBeDone, 'all' );
    for i = 1:nEval

        idx = getIndices( toDoIdx(i), self.SearchDims );
        idxC = num2cell( idx );

        try
            argsCell = namedargs2cell( self.Setups{ idxC{:} }.eval.args );
        catch
            argsCell = {};
        end

        if self.CatchErrors
            try
                thisEvaluation = ModelEvaluation( self.EvaluationNames( idxC{:} ), ...
                                                  self.Path, ...
                                                  self.Setups{ idxC{:} }, ...
                                                  argsCell{:} );
            catch ME
                warning('***** Evaluation failed *****')
                disp(['Evaluation: ' char(self.EvaluationNames(idxC{:})) ]);
                disp(['Error Message: ' ME.message]);
                for k = 1:length(ME.stack)
                    disp([ME.stack(k).name ', (line ' ...
                                         num2str(ME.stack(k).line) ')']);
                end
                self.ErrorMessages(i) = ME.message;
                
                return
            end
        
        else
            thisEvaluation = ModelEvaluation( self.EvaluationNames( idxC{:} ), ...
                                              self.Path, ...
                                              self.Setups{ idxC{:} }, ...
                                              argsCell{:} );

        end

        % save the evaluations
        thisEvaluation.conserveMemory( self.MemorySaving );
        thisEvaluation.save;

        % record results
        self.Evaluations{ idxC{:} } = thisEvaluation;
        self.IsComplete(i) = true;
        try
            self.logResults( idxC, size(self.Evaluations) );
        catch
            disp(['Unable to log results for ' char(self.EvaluationNames(idxC{:}))]);
        end

    end
    
end   