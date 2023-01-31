function self = evaluate( self, thisTrnSet, thisValSet )
    % Evaluate the model with a specified dataset
    % It may be a full or compact model
    arguments
        self            RepresentationModel
        thisTrnSet      ModelDataset
        thisValSet      ModelDataset
    end

    [ self.Loss.Training, ...
        self.Predictions.Training, ...
            self.Correlations.Training ] = ...
                        self.evaluateSet( self, thisTrnSet );

    if thisValSet.NumObs > 0
        [ self.Loss.Testing, ...
            self.Predictions.Testing, ...
                self.Correlations.Testing ] = ...
                            self.evaluateSet( self, thisValSet );
    end

end