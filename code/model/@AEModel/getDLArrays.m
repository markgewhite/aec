function [ dlX, dlY, dlXN ] = getDLArrays( self, thisDataset, maxObs )
    % Convert X, Y and XN into dl arrays
    arguments
        self            AEModel
        thisDataset     ModelDataset
        maxObs          double = 0
    end
    
    X = thisDataset.XInput;
    XN = thisDataset.XTarget( self.TSpan.Target );

    dlX = dlarray( X, self.XDimLabels );
    dlXN = dlarray( XN, self.XDimLabels );
    dlY = dlarray( thisDataset.Y, 'BC' );

    % apply the cap, if specified
    if maxObs > 0
        maxObs = min( maxObs, thisDataset.NumObs );
        idx = randsample( thisDataset.NumObs, maxObs );
        if find(dims(dlX)=='B') == 2
            dlX = dlX( :, idx, : );
        else
            dlX = dlX( :, :, idx );
        end
        if find(dims(dlXN)=='B') == 2
            dlXN = dlXN( :, idx, : );
        else
            dlXN = dlXN( :, :, idx );            
        end
        if find(dims(dlY)=='B') == 1
            dlY = dlY( idx, : );
        else
            dlY = dlY( :, idx );
        end
    end

end