function conserveMemory( self, level )
    % Conserve memory usage
    arguments
        self            ModelEvaluation
        level           double ...
            {mustBeInRange( level, 0, 3 )} = 0
    end

    for k = 1:self.NumModels
        self.Models{k} = self.Models{k}.compress( level );
    end

end