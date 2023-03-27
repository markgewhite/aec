function reload( self, memorySaving )
    % Reload all evaluations and compile results
    arguments
        self            Investigation
        memorySaving    double {mustBeInteger, ...
                                mustBeInRange( memorySaving, 0, 3 )} = 1
    end

    % save the current default figure visibility setting
    visibility = get(0, 'DefaultFigureVisible');

    % set the default figure visibility to 'off'
    set(0, 'DefaultFigureVisible', 'off');
    
    for i = 1:self.NumEvaluations

        idx = getIndices( i, self.SearchDims );
        idxC = num2cell( idx );

        % load the evaluation
        filename = strcat( self.EvaluationNames( idxC{:} ), "-Evaluation" );
        disp(['Loading ' char(filename)]);
        load( fullfile( self.Path, filename ), 'thisEvaluation' );

        thisEvaluation.conserveMemory( memorySaving );

        % record results
        self.Evaluations{ idxC{:} } = thisEvaluation;
        self.logResults( idxC, size(self.Evaluations) );

    end

    % reset the default figure visibility back to the original setting
    set(0, 'DefaultFigureVisible', visibility);

end

