function isFixed = isFixedLength( self )
    % return whether data is time-normalized
    arguments
        self    ModelDataset
    end

    isFixed = self.HasNormalizedInput;
    
end