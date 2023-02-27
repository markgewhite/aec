function [ XC, XMean, offsets ] = calcLatentComponents( self, dlZ, args )
    % Calculate the funtional components using the response function
    arguments
        self                RepresentationModel
        dlZ                 {mustBeA( dlZ, {'dlarray', 'double'} )}
        args.dlXC           {mustBeA( args.dlXC, {'dlarray', 'double'} )} = []
        args.maxObs         double {mustBeInteger} = 500
        args.responseFcn    function_handle
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Regular', 'Component'} )} = 'Component' 
        args.nSample        double {mustBeInteger} = 20
    end

    if isfield( args, 'responseFcn' ) 
        thisResponseFcn = args.responseFcn;
    else
        thisResponseFcn = self.LatentResponseFcn;
    end

    [XC, offsets ] = self.calcResponse( dlZ, ...
                                        sampling = args.sampling, ...
                                        nSample = args.nSample, ...
                                        modelFcn = thisResponseFcn, ...
                                        maxObs = args.maxObs );

    % put XC into the appropriate structure
    % Points, Samples, Components, Channels
    nSamples = size(XC, 2);

    % extract the mean curve based on Z
    XMean = XC( :, ceil(nSamples/2), :, : );
    
    switch self.ComponentCentering
        case 'Z'
            % centre about the curve generated by mean Z
            XC = XC - XMean;
        case 'X'
            % centre about the mean generated curve
            XC = XC - mean( XC, length(size(XC)) );
    end

end
