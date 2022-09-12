function save( self, path, name )
    % Save the evaluation to a specified path
    arguments
        self        ModelEvaluation
        path        string {mustBeFolder}
        name        string
    end

    % define a small structure for saving
    output.BespokeSetup = self.BespokeSetup;
    output.CVComponents = self.CVComponents;
    output.CVAuxMetrics = self.CVAuxMetrics;
    output.CVLoss = self.CVLoss;
    output.CVCorrelations = self.CVCorrelations;
    
    name = strcat( name, "-OverallEvaluation" );
    save( fullfile( path, name ), 'output' );

end   
