function [ eval, pred, cor ] = evaluateSet( self, thisDataset )
    % Evaluate the model with a specified dataset
    % doing additional work to the superclass method
    arguments
        self             SubAEModel
        thisDataset      ModelDataset
    end

    % call the superclass method
    [ eval, pred, cor ] = evaluateSet@SubRepresentationModel( self, thisDataset );

    if any(self.LossFcnTbl.Types == 'Comparator')
        % compute the comparator loss using the comparator network
        pred.ComparatorYHat = predictCompNet( self, thisDataset );
        eval.Comparator = evaluateClassifier( pred.Y, pred.ComparatorYHat );
    end

    if any(self.LossFcnTbl.Types == 'Auxiliary')
        % compute the auxiliary loss using the network
        pred.AuxNetworkYHat = predictAuxNet( self, pred.Z )';
        eval.AuxNetwork = evaluateClassifier( pred.Y, pred.AuxNetworkYHat );
    end

    eval = flattenStruct( eval );

end