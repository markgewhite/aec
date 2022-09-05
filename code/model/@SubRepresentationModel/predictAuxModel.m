function [ YHat, YHatScore] = predictAuxModel( self, Z )
    % Make prediction from Z using an auxiliary model
    arguments
        self            SubRepresentationModel
        Z               {mustBeA(Z, {'double', 'dlarray'})}
    end

    if isa( Z, 'dlarray' )
        Z = double(extractdata(Z));
    end

    doTranspose = (size(Z,2) ~= self.ZDim);
    if doTranspose
        Z = Z';
    end

    [YHat, YHatScore] = predict( self.AuxModel, Z );

    if doTranspose
        YHat = YHat';
        YHatScore = YHatScore';
    end

end