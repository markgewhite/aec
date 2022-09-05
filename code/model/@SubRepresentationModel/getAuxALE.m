function [auxALE, Q] = getAuxALE( self, thisDataset, args )
    % Generate the model's partial dependence to latent codes
    arguments
        self            SubRepresentationModel
        thisDataset     ModelDataset
        args.nSample    double {mustBeInteger} = 20
        args.auxFcn     function_handle = @predictAuxModel
    end

    % generate the latent encodings
    Z = self.encode( thisDataset );

    % define the query points by z-scores
    [auxALE, Q] = self.calcALE( Z, ...
                      sampling = 'Regular', ...
                      modelFcn = args.auxFcn, ...
                      nSample = args.nSample, ...
                      maxObs = 10000 );

end