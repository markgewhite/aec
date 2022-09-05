function plotModel( self )
    % Display the full model plots
    arguments
        self        ModelEvaluation
    end

    if ~empty( self.Model.Figs ) && ~empty( self.Model.Axes )

        % plot latent space
        plotZDist( self.Model, self.TestingPredictions.Z );
        plotZClusters( self.Model, self.TestingPredictions.Z, ...
                                   Y = self.TestingDataset.Y );
        % plot the components
        plotLatentComp( self.Model, type = 'Smoothed', shading = true );
    
    else
        % graphics objects must have been cleared
        eid = 'Evaluation:NoGrahicsObjects';
        msg = 'There are no graphics objects specified in the model.';
        throwAsCaller( MException(eid,msg) );

    end

end