function save( self )
    % Save the model plots and the object itself
    arguments
        self            RepresentationModel
    end

    plotObjects = self.Axes;
    plotObjects.Components = self.Figs.Components;
    savePlots( plotObjects, self.Info.Path, self.Info.Name );

end