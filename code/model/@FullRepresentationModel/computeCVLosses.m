function self = computeCVLosses( self )
    % Calculate the cross-validated losses
    % using predictions from the sub-models
    arguments
        self        FullRepresentationModel
    end

    self.Loss.Training = collateLosses( self.SubModels, 'Training' );
    self.CVLoss.Training = calcCVLoss( self.SubModels, 'Training' );

    if isfield( self.SubModels{1}.Loss, 'Validation' )
        self.Loss.Validation = collateLosses( self.SubModels, 'Validation' );
        self.CVLoss.Validation = calcCVLoss( self.SubModels, 'Validation' );
    end

end