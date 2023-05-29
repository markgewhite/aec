function [self, thisDataset] = finalizeInit( self, thisDataset )
    % Post-construction model initialization
    % For tasks that depend on subclass initiation 
    % but apply to all models
    arguments
        self           RepresentationModel
        thisDataset    ModelDataset
    end

    self = self.setXTargetDim;
    thisDataset = thisDataset.initTarget( self.XTargetDim );

    self.TSpan.Target = thisDataset.TSpan.Target;
    self.FDA.FdParamsTarget = thisDataset.FDA.FdParamsTarget;

end