function evaluateAll( self )
    % Recompute all evaluations
    arguments
        self                Investigation
    end

    % reset completions list
    self.IsComplete = false( self.NumEvaluations, 1 );
    for i = 1:self.NumEvaluations

        idx = getIndices( i, self.SearchDims );
        idxC = num2cell( idx );
        thisEvaluation = self.Evaluations{ idxC{:} };

        for k = 1:thisEvaluation.NumModels
            thisEvaluation.Models{k} = thisEvaluation.Models{k}.evaluate( thisEvaluation.TrainingDataset, ...
                                               thisEvaluation.TestingDataset );
        end

        thisEvaluation.evaluateModels( 'Training' );
        thisEvaluation.evaluateModels( 'Testing' );

        self.logResults( idxC, size(self.Setups) );

    end



end

