function self = plotAllALE( self )
    % Plot all the ALE curves from the sub-models (override)
    arguments
        self            FullAEModel
    end

    plotAllALE@FullRepresentationModel( self, type = 'Model' );

    if any(self.LossFcnTbl.Types == 'Auxiliary')
        plotAllALE@FullRepresentationModel( self, type = 'Network' );
    end

end