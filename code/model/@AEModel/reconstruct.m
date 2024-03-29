function [ XHat, XHatSmth ] = reconstruct( self, Z, args )
    % Reconstruct X from Z using the model
    arguments
        self            AEModel
        Z               {mustBeA(Z, {'double', 'single', 'dlarray'})}
        args.centre     logical = true
        args.convert    logical = true
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

    if args.convert
        % convert from dlarray
        XHat = double(extractdata(dlXHat));
        XHat = squeeze(permute( XHat, [1 3 2] ));

        if args.smooth   
            XHatSmth = smoothSeries( XHat, ...
                                 self.TSpan.Target, ...
                                 self.TSpan.Input, ...
                                 self.FDA.FdParamsInput );
        else
            XHatSmth = [];
        end

    else
        XHat = dlXHat;
        XHatSmth = [];

    end

end