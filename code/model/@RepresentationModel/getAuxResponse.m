function self = getAuxResponse( self, thisDataset, args )
    % Generate the model's response to latent codes
    arguments
        self            RepresentationModel
        thisDataset     ModelDataset
        args.nSample    double {mustBeInteger} = 20
    end

    % generate the latent encodings
    Z = self.encode( thisDataset );

    % define the query points by z-scores
    [self.AuxModelResponse, self.ResponseQuantiles] = ...
                                self.calcALE( Z, ...
                                              sampling = 'Regular', ...
                                              modelFcn = @predictAuxModel, ...
                                              nSample = args.nSample, ...
                                              maxObs = 10000 );

end