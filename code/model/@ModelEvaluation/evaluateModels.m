function self = evaluateModels( self, set )
    % Evaluate the trained models
    arguments
        self            ModelEvaluation
        set             char ...
            {mustBeMember( set, {'Training', 'Validation'} )}
    end

    self.CVLoss.(set) = calcCVParameters( self.Models, 'Loss', set );

    self.CVLoss.(set).Aggregated = calcCVLoss( self.Models, set );

    self.CVCorrelations.(set) = calcCVParameters( self.Models, 'Correlations', set );
   
end