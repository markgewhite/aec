function [ dlX, dlY, dlXN ] = getDLArrays( self, thisDataset, maxObs )
    % Convert X, Y and XN into dl arrays
    arguments
        self            AEModel
        thisDataset     ModelDataset
        maxObs          double = 0
    end
    
    if self.UsesFdCoefficients
        X = thisDataset.XInputCoeffRegular;
        XN = thisDataset.XTargetCoeff;
    else
        X = thisDataset.XInputRegular;
        XN = thisDataset.XTarget;
    end

    dlX = dlarray( X, self.XDimLabels );

    if length(size( XN, 3) )==3
        XN = permute( XN, [1 3 2] );
    end
    dlXN = dlarray( XN, self.XDimLabels );

    dlY = dlarray( thisDataset.Y, 'CB' );

    % apply the cap, if specified
    if maxObs > 0
        maxObs = min( maxObs, thisDataset.NumObs );
        idx = randsample( length(dlY), maxObs );
        dlX = dlX( :, idx );
        dlXN = dlXN( :, idx );
        dlY = dlY( idx );
    end

end