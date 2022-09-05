function self = incrementActiveZDim( self )
    % Increment the number of active dimensions
    arguments
        self            SubAEModel
    end

    self.ZDimActive = min( self.ZDimActive + 1, self.ZDim );

end