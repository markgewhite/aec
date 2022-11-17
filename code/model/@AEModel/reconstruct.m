function [ dlXHat, XHatSmth, XHatReg ] = reconstruct( self, Z, args )
    % Reconstruct X from Z using the model
    arguments
        self            AEModel
        Z               {mustBeA(Z, {'double', 'dlarray'})}
        args.convert    logical = true
    end

    if isa( Z, 'dlarray' )
        dlZ = Z;
    else
        dlZ = dlarray( Z', 'CB' );
    end

    dlXHat = decodeDispatcher( self, dlZ, forward = false );

    if self.HasCentredDecoder
        dlXHat = dlXHat + self.MeanCurveTarget;
    end

    XHatSmth = [];
    XHatReg = [];

    if args.convert

        if isa( dlXHat, 'dlarray' )
            dlXHat = double(extractdata(gather(dlXHat)));
        end
        dlXHat = permute( dlXHat, [1 3 2] );
        dlXHat = squeeze( dlXHat ); 

        XHatSmth = smoothSeries(  dlXHat, ...
                                  self.TSpan.Target, ...
                                  self.TSpan.Target, ...
                                  self.FDA.FdParamsTarget );
        XHatReg = smoothSeries(   dlXHat, ...
                                  self.TSpan.Target, ...
                                  self.TSpan.Regular, ...
                                  self.FDA.FdParamsTarget );

    end

end