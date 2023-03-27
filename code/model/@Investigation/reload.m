function reload( self )
    % Reload all evaluations and compile results
    arguments
        self            Investigation
    end

    for i = 1:self.NumEvaluations

        idx = getIndices( i, self.SearchDims );
        idxC = num2cell( idx );

        % load the evaluation
        filename = strcat( self.EvaluationNames( idxC{:} ), "-Evaluation" );
        disp(['Loading ' char(filename)]);
        load( fullfile( self.Path, filename ), 'thisEvaluation' );

        % record results
        self.Evaluations{ idxC{:} } = thisEvaluation;
        self.logResults( idxC, size(self.Evaluations) );

    end

end

