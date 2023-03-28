function reload( self, args )
    % Reload all evaluations and compile results
    arguments
        self                Investigation
        args.MemorySaving   double {mustBeInteger, ...
                        mustBeInRange( args.MemorySaving, 0, 3 )} = 1
        args.Overwrite      logical = true
    end

    % save the current default figure visibility setting
    visibility = get(0, 'DefaultFigureVisible');

    % set the default figure visibility to 'off'
    set(0, 'DefaultFigureVisible', 'off');
    
    for i = 1:self.NumEvaluations

        idx = getIndices( i, self.SearchDims );
        idxC = num2cell( idx );

        if args.Overwrite || isempty(self.Evaluations{ idxC{:} })
            % load the evaluation
            filename = strcat( self.EvaluationNames( idxC{:} ), "-Evaluation" );
            disp(['Loading ' char(filename)]);
            load( fullfile( self.Path, filename ), 'thisEvaluation' );
    
            thisEvaluation.conserveMemory( args.MemorySaving );
    
            self.Evaluations{ idxC{:} } = thisEvaluation;
        end

        self.logResults( idxC, size(self.Evaluations) );

    end

    % reset the default figure visibility back to the original setting
    set(0, 'DefaultFigureVisible', visibility);

end

