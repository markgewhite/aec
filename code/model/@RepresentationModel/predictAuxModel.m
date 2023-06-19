function [ YHat, YHatScore] = predictAuxModel( self, Z )
    % Make prediction from Z using an auxiliary model
    arguments
        self            RepresentationModel
        Z               double
    end

    if strcmp( self.AuxObjective, 'Classification' )
        [YHat, YHatScore] = predict( self.AuxModel, Z );
    else
        YHat = predict( self.AuxModel, Z );
        YHatScore = [];
    end

end