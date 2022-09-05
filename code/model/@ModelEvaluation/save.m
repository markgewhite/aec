function save( self, path, name )
    % Save the evaluation to a specified path
    arguments
        self        ModelEvaluation
        path        string {mustBeFolder}
        name        string
    end

    % define a small structure for saving
    output.BespokeSetup = self.BespokeSetup;
    output.TrainingEvaluation = self.TrainingEvaluation;
    output.TestingEvaluation = self.TestingEvaluation;
    output.TrainingCorrelations = self.TrainingCorrelations;
    output.TestingCorrelations = self.TestingCorrelations;
    
    name = strcat( name, "-OverallEvaluation" );
    save( fullfile( path, name ), 'output' );

end   