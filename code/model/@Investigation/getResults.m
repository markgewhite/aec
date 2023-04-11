function report = getResults( self )
    % Return a structure summarizing the results
    arguments
        self        Investigation
    end

    % define a small structure for saving
    report.Name = self.Name;
    report.Path = self.Path;
    report.BaselineSetup = self.BaselineSetup;
    report.Parameters = self.Parameters;
    report.GridSearch = self.GridSearch;
    report.IsComplete = self.IsComplete;
    report.TrainingResults = self.TrainingResults;
    report.TestingResults = self.TestingResults;

end