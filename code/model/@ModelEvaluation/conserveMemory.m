function conserveMemory( self, level, closeFigs )
    % Conserve memory usage
    arguments
        self            ModelEvaluation
        level           double ...
            {mustBeInRange( level, 0, 3 )} = 0
        closeFigs       logical = false
    end

    for k = 1:self.NumModels
        if closeFigs
            self.Models{k}.closeFigures;
        end
        self.Models{k} = self.Models{k}.compress( level );
    end

end