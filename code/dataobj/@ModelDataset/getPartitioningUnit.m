function unit = getPartitioningUnit( self )
    % Provide the variable to be unit for partitioning
    % Placeholder function that may be overridden by children
    arguments
        self    ModelDataset
    end

    unit = 1:self.NumObs;

end