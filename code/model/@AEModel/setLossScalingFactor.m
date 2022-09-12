function self = setLossScalingFactor( self )
    % Set the scaling factors for reconstructions
    arguments
        self            AEModel
    end
    
    for i = 1:size( self.LossFcnTbl, 1 )
        
        if ismember( self.LossFcnTbl.Inputs(i), {'X-XHat', 'XC', 'XHat'} )
            name = self.LossFcnTbl.Names(i);
            self.LossFcns.(name).Scale = self.Scale;
        end

    end

end