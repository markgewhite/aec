function [ YHatFold, YHatMaj ] = predictCompNet( self, thisDataset )
    % Predict Y from X using all comparator networks
    arguments
        self            FullAEModel
        thisDataset     ModelDataset
    end

    YHatFold = zeros( thisDataset.NumObs, self.KFolds );
    for k = 1:self.KFolds
        YHatFold( :, k ) = predictCompNet( self.SubModels{k}, thisDataset );
    end

    YHatMaj = zeros( thisDataset.NumObs, 1 );
    for i = 1:thisDataset.NumObs
        [votes, grps] = groupcounts( YHatFold(i,:)' );
        [ ~, idx ] = max( votes );
        YHatMaj(i) = grps( idx );
    end

end