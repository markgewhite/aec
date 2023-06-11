function [ YHat, YHatScore] = predictAuxModel( self, Z )
    % Make prediction from Z using an auxiliary model
    arguments
        self            RepresentationModel
        Z               {mustBeA(Z, {'double', 'single', 'dlarray'})}
    end

    if isa( Z, 'dlarray' )
        Z = double(extractdata(gather(Z)));
    end

    doTranspose = (size(Z,2) ~= self.ZDim);
    if doTranspose
        Z = Z';
    end

    if strcmp( self.AuxObjective, 'Classification' )
        [YHat, YHatScore] = predict( self.AuxModel, Z );
    else
        YHat = predict( self.AuxModel, Z );
        YHatScore = [];
    end

    if doTranspose
        YHat = YHat';
        YHatScore = YHatScore';
    end

end