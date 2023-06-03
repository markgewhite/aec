function [ dlXC, Q, dlZC ] = calcLatentComponents( self, dlZ, args )
    % Calculate the funtional components using the response function
    arguments
        self                RepresentationModel
        dlZ                 {mustBeA( dlZ, {'dlarray', 'double'} )}
        args.dlXB           {mustBeA( args.dlXB, 'cell' )}
        args.dlXC           {mustBeA( args.dlXC, {'dlarray', 'double'} )}
        args.mode           char ...
                            {mustBeMember(args.mode, ...
                            {'Full', 'InputOnly', 'OutputOnly'} )} = 'Full' 
        args.maxObs         double {mustBeInteger} = 500
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Regular', 'Component'} )} = 'Component' 
        args.nSample        double {mustBeInteger} = 20
    end

    % calculate the response of the specified type
    argsCell = namedargs2cell( args );
    switch self.ComponentType
        case 'ALE'
            [dlXC, Q, dlZC] = calcALE( self, dlZ, argsCell{:} );
        case {'PDP', 'FPC'}
            [dlXC, Q, dlZC] = calcPDP( self, dlZ, argsCell{:} );
        case 'AEC'
            [dlXC, Q] = calcAEC( self, dlZ, argsCell{:} );
            dlZC = [];
        otherwise
            dlXC = [];
            Q = [];
            dlZC = [];
    end

    % put XC into the appropriate structure
    % Points, Samples, Components, Channels
    nSamples = size(dlXC, 2);

    % extract the mean curve based on Z
    XMean = dlXC( :, ceil(nSamples/2), :, : );
    
    switch self.ComponentCentering
        case 'Z'
            % centre about the curve generated by mean Z
            dlXC = dlXC - XMean;
        case 'X'
            % centre about the mean generated curve
            dlXC = dlXC - mean( dlXC, length(size(dlXC)) );
    end

end
