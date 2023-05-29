function [self, thisDataset] = finalizeInit( self, thisDataset )
    % Post-construction model initialization
    % For tasks that depend on subclass initiation 
    % but apply to all models
    arguments
        self           RepresentationModel
        thisDataset    ModelDataset
    end

    self = self.setXTargetDim;

    self.TSpan.Target =  linspace( self.TSpan.Original(1), ...
                                   self.TSpan.Original(end), ...
                                   self.XTargetDim );

    self.FDA.FdParamsTarget = thisDataset.setFDAParameters( self.TSpan.Target );

end