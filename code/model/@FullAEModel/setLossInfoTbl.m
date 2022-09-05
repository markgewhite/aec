function self = setLossInfoTbl( self )
    % Update the info table
    
    nFcns = length( self.LossFcnNames );
    Names = strings( nFcns, 1 );
    Types = strings( nFcns, 1 );
    Inputs = strings( nFcns, 1 );
    Weights = zeros( nFcns, 1 );
    NumLosses = zeros( nFcns, 1 );
    LossNets = strings( nFcns, 1 );
    HasNetwork = false( nFcns, 1 );
    DoCalcLoss = false( nFcns, 1 );
    UseLoss = false( nFcns, 1 );

    for i = 1:nFcns
        
        thisLossFcn = self.LossFcns.(self.LossFcnNames(i));
        Names(i) = thisLossFcn.Name;
        Types(i) = thisLossFcn.Type;
        Inputs(i) = thisLossFcn.Input;
        Weights(i) = self.LossFcnWeights(i);
        NumLosses(i) = thisLossFcn.NumLoss;
        HasNetwork(i) = thisLossFcn.HasNetwork;
        DoCalcLoss(i) = thisLossFcn.DoCalcLoss;
        UseLoss(i) = thisLossFcn.UseLoss;

        nFcnNets = length( thisLossFcn.LossNets );
        for j = 1:nFcnNets
            if length(string( thisLossFcn.LossNets{j} ))==1
                assignments = thisLossFcn.LossNets{j};
            else
                assignments = strjoin( thisLossFcn.LossNets{j,:}, '+' );
            end
            LossNets(i) = strcat( LossNets(i), assignments ) ;
            if j < nFcnNets
                LossNets(i) = strcat( LossNets(i), "; " );
            end
        end

    end

    self.LossFcnTbl = table( Names, Types, Inputs, Weights, ...
            NumLosses, LossNets, HasNetwork, DoCalcLoss, UseLoss );

end
