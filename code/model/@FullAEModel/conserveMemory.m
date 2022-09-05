function self = conserveMemory( self, level )
    % Conserve memory usage for AE
    arguments
        self            FullAEModel
        level           double ...
            {mustBeInRange( level, 0, 4 )} = 0
    end

    self = conserveMemory@FullRepresentationModel( self, level );

    if level >= 3
        for k = 1:self.KFolds
            self.SubModels{k}.Optimizer = [];
        end
    end

end