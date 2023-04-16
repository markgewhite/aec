function self = getAuxResponse( self, thisDataset, args )
    % Generate the model's response to latent codes
    arguments
        self            RepresentationModel
        thisDataset     ModelDataset
        args.nSample    double {mustBeInteger} = 20
    end

    % generate the latent encodings
    Z = self.encode( thisDataset, auxiliary = true );

    % define the query points by z-scores
    thisResponseFcn = @(Z) predictAuxModel( self, Z );
    [self.AuxModelResponse, self.ResponseQuantiles] = ...
                                self.calcResponse( Z, ...
                                              sampling = 'Regular', ...
                                              modelFcn = thisResponseFcn, ...
                                              nSample = args.nSample, ...
                                              maxObs = 10000 );

end