function save( self )
    % Save the model plots and the object itself
    arguments
        self            AEModel
    end

    if self.ShowPlots
        plotObjects = self.Axes;
        plotObjects.Components = self.Figs.Components;
        plotObjects.LossFig = self.Trainer.LossFig;

        savePlots( plotObjects, self.Info.Path, self.Info.Name );
    end
    
end