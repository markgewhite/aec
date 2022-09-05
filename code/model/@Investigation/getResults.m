function report = getResults( self )
    % Return a structure summarizing the results
    arguments
        self        Investigation
    end

    % define a small structure for saving
    report.BaselineSetup = self.BaselineSetup;
    report.GridSearch = self.GridSearch;
    report.TrainingResults = self.TrainingResults;
    report.TestingResults = self.TestingResults;

end