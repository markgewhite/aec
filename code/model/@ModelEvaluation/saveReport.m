function saveReport( self )
    % Save the evaluation report to a specified path
    arguments
        self        ModelEvaluation
    end

    % define a small structure for saving
    eval.BespokeSetup = self.BespokeSetup;
    eval.CVComponents = self.CVComponents;
    eval.CVAuxMetrics = self.CVAuxMetrics;
    eval.CVLoss = self.CVLoss;
    eval.CVCorrelations = self.CVCorrelations;
    
    filename = strcat( self.Name, "-EvaluationReport" );
    save( fullfile( self.Path, filename ), 'eval' );

end   