function thisSubset = partition( self, idx )
    % Create the subset of this ModelDataset
    % using the indices specified
    arguments
        self        ModelDataset
        idx         logical 
    end

    thisSubset = self;

    thisSubset.XFd = splitFd( self.XFd, idx );
    thisSubset.XLen = self.XLen( idx );
    thisSubset.Y = self.Y( idx );
    thisSubset.SubjectID = self.SubjectID( idx );

end
