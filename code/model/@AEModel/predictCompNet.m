function YHat = predictCompNet( self, thisDataset )
    % Make prediction from X using the comparator network
    arguments
        self            AEModel
        thisDataset     ModelDataset
    end

    dlX = self.getDLArrays( thisDataset );

    if self.FlattenInput && size( dlX, 3 ) > 1
        dlX = flattenDLArray( dlX );
    end

    compNet = (self.LossFcnTbl.Types == 'Comparator');
    if ~any( compNet )
        eid = 'aeModel:NoComparatorFunction';
        msg = 'No comparator loss function specified in the model.';
        throwAsCaller( MException(eid,msg) );
    end

    if ~self.LossFcnTbl.HasNetwork( compNet )
        eid = 'aeModel:NoComparatorNetwork';
        msg = 'No comparator network specified in the model.';
        throwAsCaller( MException(eid,msg) );
    end

    compNetName = self.LossFcnTbl.Names( compNet );

    dlYHat = predict( self.Nets.(compNetName), dlX );

    YHat = double(extractdata( dlYHat ))';
    YHat = double(onehotdecode( YHat, 1:thisDataset.CDim, 2 ));

end