function self = compress( self, level )
    % Clear the objects to save memory
    arguments
        self            SubRepresentationModel
        level           double {mustBeInRange( level, 0, 3 )} = 0
    end

    if level >= 1
        self.Figs = [];
        self.Axes = [];
    end

    if level >= 2
        self.Predictions = [];
    end

end