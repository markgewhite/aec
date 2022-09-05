function self = evaluate( self, thisTrnSet, thisValSet )
    % Evaluate the model with a specified dataset
    % It may be a full or compact model
    arguments
        self            SubRepresentationModel
        thisTrnSet      ModelDataset
        thisValSet      ModelDataset
    end

    [ self.Loss.Training, ...
        self.Predictions.Training, ...
            self.Correlations.Training ] = ...
                        self.evaluateSet( self, thisTrnSet );

    if thisValSet.NumObs > 0
        [ self.Loss.Validation, ...
            self.Predictions.Validation, ...
                self.Correlations.Validation ] = ...
                            self.evaluateSet( self, thisValSet );
    end

end