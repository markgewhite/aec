function self = conserveMemory( self, level )
    % Conserve memory usage
    arguments
        self            FullRepresentationModel
        level           double ...
            {mustBeInRange( level, 0, 3 )} = 0
    end

    if level >= 1
        self.Figs = [];
        self.Axes = [];
    end

    for k = 1:self.KFolds
        self.SubModels{k} = ...
            self.SubModels{k}.compress( self.CompressionLevel );
    end

end