function [F, Q, Z, offsets] = calcResponse( self, dlZ, args )
    % Call the relevant response function
    % for latent component generation and the auxiliary model
    arguments
        self                RepresentationModel
        dlZ                 {mustBeA( dlZ, {'dlarray', 'double'} )}
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Regular', 'Component'} )} = 'Regular'
        args.nSample        double {mustBeInteger} = 20
        args.maxObs         double = 1000
        args.modelFcn       function_handle
        args.modelFcnArgs   cell = []
    end
    
    % calculate the response of the specified type
    argsCell = namedargs2cell( args );
    switch self.ComponentType
        case {'ALE', 'FPC'}
            [F, Q, Z, offsets] = calcALE( self, dlZ, argsCell{:} );
        otherwise
            F = [];
            Q = [];
            Z = [];
            offsets = [];
    end

end
