function [ YHat, YHatScore] = predictAuxModel( self, Z )
    % Make prediction from Z using an auxiliary model
    arguments
        self            AEModel
        Z               {mustBeA(Z, {'double', 'single', 'dlarray'})}
    end

    if isa( Z, 'dlarray' )
        Z = double(extractdata(Z));
    end

    doTranspose = (size(Z,2) ~= self.ZDim);
    if doTranspose
        Z = Z';
    end

    Z = Z( :, 1:self.ZDimAux );

    [YHat, YHatScore] = predictAuxModel@AEModel( self, Z );
    
    if doTranspose
        YHat = YHat';
        YHatScore = YHatScore';
    end

end