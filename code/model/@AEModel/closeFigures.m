function closeFigures( self )
    % Override - close all figures including the trainer's
    arguments
        self            AEModel
    end

    closeFigures@RepresentationModel( self );

    try
        close( self.Trainer.LossFig );
    catch
        disp('Could not close figure = LossFig');
    end

end