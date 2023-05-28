function self = setLossScalingFactor( self )
    % Set the scaling factors for reconstructions
    arguments
        self            AEModel
    end
    
    for i = 1:size( self.LossFcnTbl, 1 )
        
        if any(ismember( self.LossFcnTbl.Inputs{i}, {'dlXIn', 'dlXOut', 'dlXHat', 'dlXC', 'dlXGen'} ))
            name = self.LossFcnTbl.Names(i);
            self.LossFcns.(name).Scale = self.Scale;
        end

    end

end