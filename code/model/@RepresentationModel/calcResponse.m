function [F, Q, Z] = calcResponse( self, dlZ, args )
    % Call the relevant response function
    % for latent component generation and the auxiliary model
    arguments
        self                RepresentationModel
        dlZ                 {mustBeA( dlZ, {'dlarray', 'double'} )}
        args.mode           char ...
                            {mustBeMember(args.mode, ...
                            {'Full', 'InputOnly', 'OutputOnly'} )} = 'Full' 
        args.dlXC           {mustBeA( args.dlXC, {'dlarray', 'double'} )} = []
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Regular', 'Component'} )} = 'Regular'
        args.nSample        double {mustBeInteger} = 20
        args.maxObs         double = 1000
        args.modelFcn       function_handle
    end
    
    % calculate the response of the specified type
    argsCell = namedargs2cell( args );
    switch self.ComponentType
        case 'ALE'
            [F, Q, Z] = calcALE( self, dlZ, argsCell{:} );
        case {'PDP', 'FPC'}
            [F, Q, Z] = calcPDP( self, dlZ, argsCell{:} );
        case 'AEC'
            [F, Q, Z] = calcAEC( self, dlZ, argsCell{:} );
        otherwise
            F = [];
            Q = [];
            Z = [];
    end

end
