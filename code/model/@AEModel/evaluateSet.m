function [ eval, pred, cor ] = evaluateSet( self, thisDataset )
    % Evaluate the model with a specified dataset
    % doing additional work to the superclass method
    arguments
        self             AEModel
        thisDataset      ModelDataset
    end

    % call the superclass method
    [ eval, pred, cor ] = evaluateSet@RepresentationModel( self, thisDataset );

    if any(self.LossFcnTbl.Types == 'Comparator')
        % compute the comparator loss using the comparator network
        pred.ComparatorYHat = predictCompNet( self, thisDataset );
        eval.Comparator = evaluateClassifier( pred.Y, pred.ComparatorYHat );
    end

    if any(self.LossFcnTbl.Types == 'Auxiliary')
        % compute the auxiliary loss using the network
        pred.AuxNetworkYHat = predictAuxNet( self, pred.Z )';
        switch self.AuxObjective
            case 'Classification'
                eval.AuxNetwork = evaluateClassifier( pred.Y, pred.AuxNetworkYHat );
            case 'Regression'
                eval.AuxNetwork = evaluateRegressor( pred.Y, pred.AuxNetworkYHat );
        end
    end

    eval = flattenStruct( eval );

end