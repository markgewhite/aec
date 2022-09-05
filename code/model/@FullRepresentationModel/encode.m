function [ ZFold, ZMean, ZSD ] = encode( self, thisDataset )
    % Encode aggregated features Z from X using all models
    arguments
        self            FullRepresentationModel
        thisDataset     ModelDataset
    end

    ZFold = zeros( thisDataset.NumObs, self.ZDim, self.KFolds );
    for k = 1:self.KFolds
        ZFold( :, :, k ) = encode( self.SubModels{k}, thisDataset );
    end
    ZMean = mean( ZFold, 3 );
    ZSD = std( ZFold, [], 3 );

end