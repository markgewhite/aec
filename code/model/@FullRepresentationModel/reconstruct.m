function [ XHatFold, XHatMean, XHatSD ] = reconstruct( self, Z )
    % Reconstruct aggregated X from Z using all models
    arguments
        self            FullRepresentationModel
        Z               double
    end

    isEnsemble = (size( Z, 3 ) > 1);
    XHatFold = zeros( self.XTargetDim, size(Z,1), self.KFolds );
    for k = 1:self.KFolds
        if isEnsemble
            XHatFold( :, :, k ) = ...
                    reconstruct( self.SubModels{k}, Z(:,:,k) );
        else
            XHatFold( :, :, k ) = ...
                    reconstruct( self.SubModels{k}, Z );
        end
    end
    XHatMean = mean( XHatFold, 3 );
    XHatSD = std( XHatFold, [], 3 );

end