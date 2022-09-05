function [ XC, XMean, offsets ] = calcLatentComponents( self, dlZ, args )
    % Calculate the funtional components using the decoder network
    % from Z code generated using Accumulated Local Estimation 
    arguments
        self            SubAEModel
        dlZ             {mustBeA( dlZ, {'dlarray', 'double'} )}
        args.maxObs     double {mustBeInteger} = 500
        args.forward    logical = false
        args.smooth     logical = false
        args.dlX        {mustBeA( args.dlX, {'dlarray', 'double'} )}
    end

    % prepare decoder dispatch arguments
    dispatchArgs.forward = args.forward;
    if isfield( args, 'dlX' )
        dispatchArgs.dlX = args.dlX;
    end
    dispatchArgsCell = namedargs2cell( dispatchArgs );

    [XC, ~, ~, offsets] = self.calcALE( dlZ, ...
                      sampling = 'Component', ...
                      modelFcn = @decodeDispatcher, ...
                      modelFcnArgs = dispatchArgsCell, ...
                      maxObs = args.maxObs );

    % put XC into the appropriate structure
    % Points, Samples, Components, Channels
    XC = permute( XC, [3 2 1 4] );
    nSamples = size(XC, 2);

    % extract the mean curve based on Z
    XMean = XC( :, ceil(nSamples/2), :, : );
    
    switch self.ComponentCentring
        case 'Z'
            % centre about the curve generated by mean Z
            XC = XC - XMean;
        case 'X'
            % centre about the mean generated curve
            XC = XC - mean( XC, length(size(XC)) );
    end

    if args.smooth
        % smooth to regularly-spaced time span
        XCSmth = zeros( length(self.TSpan.Regular), ...
                        size(XC,2), size(XC,3), size(XC,4) );
        for c = 1:size(XC,4)
            XCSmth(:,:,:,c) = smoothSeries( XC(:,:,:,c), ...
                                 self.TSpan.Target, ...
                                 self.TSpan.Regular, ...
                                 self.FDA.FdParamsTarget );
        end
        XC = XCSmth;
    end

end
