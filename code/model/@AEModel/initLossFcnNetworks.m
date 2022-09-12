function self = initLossFcnNetworks( self )
    % Initialize the loss function networks
    arguments
        self        AEModel
    end

    for i = 1:length( self.LossFcnNames )
        thisName = self.LossFcnNames{i};
        thisLossFcn = self.LossFcns.(thisName);
        thisType = self.LossFcnTbl.Types(self.LossFcnTbl.Names == thisName);
        if thisLossFcn.HasNetwork
            if thisType == 'Comparator' %#ok<BDSCA> 
                self.Nets.(thisName) = thisLossFcn.initNetwork( ...
                                        self.Nets.Encoder );
            else
                self.Nets.(thisName) = thisLossFcn.initNetwork;
            end
        end
    end

end