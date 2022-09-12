function self = compress( self, level )
    % Clear the objects to save memory
    % including object specific to an AE
    arguments
        self            AEModel
        level           double ...
            {mustBeInRange( level, 0, 3 )} = 0
    end

    self = compress@SubRepresentationModel( self, level );

    if level >= 1
        self.Trainer.LossFig = [];
        self.Trainer.LossLines = [];
    end

    if level == 3
        self.Optimizer = [];
    end

end