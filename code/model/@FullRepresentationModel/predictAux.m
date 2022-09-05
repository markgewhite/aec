function [ YHatFold, YHatMaj ] = predictAux( self, Z )
    % Predict Y from Z using all auxiliary models
    arguments
        self            FullRepresentationModel
        Z               double
    end

    isEnsemble = (size( Z, 3 ) > 1);
    nRows = size( Z, 1 );
    YHatFold = zeros( nRows, self.KFolds );
    for k = 1:self.KFolds
        if isEnsemble
            YHatFold( :, k ) = ...
                    predict( self.SubModels{k}.AuxModel, Z(:,:,k) );
        else
            YHatFold( :, k ) = ...
                    predict( self.SubModels{k}.AuxModel, Z );
        end
    end

    YHatMaj = zeros( nRows, 1 );
    for i = 1:nRows
        [votes, grps] = groupcounts( YHatFold(i,:)' );
        [ ~, idx ] = max( votes );
        YHatMaj(i) = grps( idx );
    end

end