function self = conserveMemory( self, level )
    % Converse memory by paring back the storage of the evaluations
    % Levels: 
    % 0 = none; 
    % 1 = graphics cleared; 
    % 2 = graphics and predictions cleared;
    % 3 = graphics, predictions, and optimzer cleared
    % 4 = evaluation erased
    arguments
        self            Investigation
        level           double {mustBeInteger, ...
                                mustBeInRange( level, 0, 4 )} = 0
    end
    
    nEval = prod( self.SearchDims );
    % run the evaluation loop
    for c = 1:nEval

        idx = getIndices( c, self.SearchDims );
        idxC = num2cell( idx );

        if level == 4
            % maximum conservation: erase the evaluation
            % useful if the grid search is extensive
            self.Evaluations{ idxC{:} } = [];
        else
            % scaled memory conservation
            self.Evaluations{ idxC{:} } = ...
                self.Evaluations{ idxC{:} }.conserveMemory( level );
        end

    end

end