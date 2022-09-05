function thisSubset = partition( self, idx )
    % Create the subset of this ModelDataset
    % using the indices specified
    arguments
        self        ModelDataset
        idx         logical 
    end

    subXRaw = split( self.XInputRaw, idx );
    subY = self.Y( idx );
    subTSpan = self.TSpan.Original;

    thisSubset = ModelDataset( subXRaw, subY, subTSpan, ...
                               purpose = 'ForSubset' );

    thisSubset.XFd = splitFd( self.XFd, idx );
    thisSubset.XLen = self.XLen( idx );

    thisSubset.Normalization = self.Normalization;
    thisSubset.NormalizedPts = self.NormalizedPts;
    thisSubset.HasNormalizedInput = self.HasNormalizedInput;

    thisSubset.Padding = self.Padding;
    thisSubset.FDA = self.FDA;
    thisSubset.ResampleRate = self.ResampleRate;

    thisSubset.XInputDim = self.XInputDim;
    thisSubset.XTargetDim = self.XTargetDim;
    thisSubset.XChannels = self.XChannels;

    thisSubset.CDim = self.CDim;
    thisSubset.YLabels = self.YLabels;
    thisSubset.NumObs = sum( idx );

    thisSubset.TSpan = self.TSpan;

    thisSubset.HasMatchingOutput = self.HasMatchingOutput;
    thisSubset.HasAdaptiveTimeSpan = self.HasAdaptiveTimeSpan;
    thisSubset.AdaptiveLowerBound = self.AdaptiveLowerBound;
    thisSubset.AdaptiveUpperBound = self.AdaptiveUpperBound;

    thisSubset.Info = self.Info;

end
