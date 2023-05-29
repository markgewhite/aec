function self = initTarget( self, XDim )
    % Initialize the target data parameters 
    arguments
        self        ModelDataset
        XDim        double
    end
   
    self.XTargetDim = XDim;

    self.TSpan.Target =  linspace( self.TSpan.Original(1), ...
                                   self.TSpan.Original(end), ...
                                   XDim );

    self.FDA.FdParamsTarget = self.setFDAParameters( self.TSpan.Target );

end

