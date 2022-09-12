function self = addLossFcnNetworks( self )
    % Add one or more networks to the model
    arguments
        self        AEModel
    end

    nFcns = length( self.LossFcnNames );
    for i = 1:nFcns
        name = self.LossFcnNames(i);
        if self.LossFcns.(name).HasNetwork
            % set the data dimensions 
            self.LossFcns.(name) = setDimensions( self.LossFcns.(name), self );
            % record its name
            self.NetNames = [ string(self.NetNames) name ];
            % increment the counter
            self.NumNetworks = self.NumNetworks + 1;
        end
    end

end