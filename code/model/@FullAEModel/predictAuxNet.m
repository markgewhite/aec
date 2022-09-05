function [ YHatFold, YHatMaj ] = predictAuxNet( self, Z, Y )
    % Predict Y from Z using all trained auxiliary networks
    arguments
        self            FullAEModel
        Z               {mustBeA(Z, {'double', 'dlarray'})}
        Y               {mustBeA(Y, {'double', 'dlarray'})}
    end

    isEnsemble = (size( Z, 3 ) > 1);
    nRows = size( Z, 1 );
    YHatFold = zeros( nRows, self.KFolds );
    for k = 1:self.KFolds
        if isEnsemble
            YHatFold( :, k ) = ...
                    predictAuxNet( self.SubModels{k}, Z(:,:,k) );
        else
            YHatFold( :, k ) = ...
                    predictAuxNet( self.SubModels{k}, Z );
        end
    end

    YHatMaj = zeros( nRows, 1 );
    for i = 1:nRows
        [votes, grps] = groupcounts( YHatFold(i,:)' );
        [ ~, idx ] = max( votes );
        YHatMaj(i) = grps( idx );
    end

end