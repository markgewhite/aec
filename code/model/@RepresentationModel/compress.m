function self = compress( self, level )
    % Clear the objects to save memory
    arguments
        self            RepresentationModel
        level           double {mustBeInRange( level, 0, 3 )} = 0
    end

    if level >= 1
        self.Figs = [];
        self.Axes = [];
        self.LatentResponseFcn = [];
    end

    if level >= 2
        self.Predictions = [];
    end

end