function [ XHat, XHatSmth, XHatReg ] = reconstruct( self, Z, args )
    % Reconstruct X from Z using the model
    arguments
        self            AEModel
        Z               {mustBeA(Z, {'double', 'single', 'dlarray'})}
        args.centre     logical = true
        args.points     logical = true
        args.smooth     logical = false
    end

    if isa( Z, 'dlarray' )
        dlZ = Z;
    else
        dlZ = dlarray( Z', 'CB' );
    end

    dlXHat = predict( self.Nets.Decoder, dlZ );

    if args.centre && self.HasCentredDecoder
        dlXHat = dlXHat + self.MeanCurveTarget;
    end

    if args.points
        % convert from dlarray
        XHat = double(extractdata(gather(dlXHat)));
        XHat = squeeze(permute( XHat, [1 3 2] ));
    else
        XHat = dlXHat;
    end

    if args.smooth && args.points

        XHatSmth = smoothSeries( XHat, ...
                                 self.TSpan.Target, ...
                                 self.TSpan.Target, ...
                                 self.FDA.FdParamsTarget );
        XHatReg = smoothSeries( XHat, ...
                                self.TSpan.Target, ...
                                self.TSpan.Regular, ...
                                self.FDA.FdParamsTarget );

    else
        XHatSmth = [];
        XHatReg = [];

    end

end