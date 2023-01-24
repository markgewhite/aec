function self = getAuxResponse( self, thisDataset, args )
    % Generate the model and network's response to latent codes
    arguments
        self            AEModel
        thisDataset     ModelDataset
        args.nSample    double {mustBeInteger} = 20
    end

    % generate the latent encodings
    dlZ = self.encode( thisDataset, convert = false );
    Z = double(extractdata(gather(dlZ)))';

    % define the query points by z-scores
    [self.AuxModelResponse, self.ResponseQuantiles] = ...
                                self.calcResponse( Z, ...
                                              sampling = 'Regular', ...
                                              modelFcn = @predictAuxModel, ...
                                              nSample = args.nSample, ...
                                              maxObs = 10000 );

    % re-run for the auxiliary network, if present
    if any(self.LossFcnTbl.Types=='Auxiliary')
        self.AuxNetResponse = self.calcResponse( dlZ, ...
                                            sampling = 'Regular', ...
                                            modelFcn = @predictAuxNet, ...
                                            nSample = args.nSample, ...
                                            maxObs = 10000 );
    else
        self.AuxNetResponse = [];
    end

end