function self = load( self )
    % Load a previousinvestigation
    arguments
        self        Investigation
    end
   
    name = strcat( self.Name, "-Investigation" );
    load( fullfile( self.Path, name ), 'report' );

    self.BaselineSetup = report.BaselineSetup;
    self.Parameters = report.Parameters;
    self.GridSearch = report.GridSearch;
    self.TrainingResults = report.TrainingResults;
    self.TestingResults = report.TestingResults;

end