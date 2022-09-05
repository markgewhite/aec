function self = computeCVCorrelations( self )
    % Average the correlation matrices across sub-models
    arguments
        self        FullRepresentationModel
    end

    self.CVCorrelation.Training = ...
            calcCVCorrelations( self.SubModels, 'Training' );

    if isfield( self.SubModels{1}.Loss, 'Validation' )
        self.CVCorrelation.Validation = ...
            calcCVCorrelations( self.SubModels, 'Validation' );
    end

end