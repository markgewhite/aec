function [ XHat, XHatSmth, XHatReg ] = reconstruct( self, Z, args )
    % Reconstruct X from Z using the model
    arguments
        self            AEModel
        Z               {mustBeA(Z, {'double', 'dlarray'})}
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
    
    if self.UsesFdCoefficients && args.points
        % convert to real points

        % create a dummy Fd object
        dummy = zeros( length(self.TSpan.Target), size(XHat,2), size(XHat,3) );
        XFd = smooth_basis( self.TSpan.Target, ...
                            dummy, ...
                            self.FDA.FdParamsTarget );

        % impose the coefficient matrix
        XFd = putcoef( XFd, XHat );

        % evaluate the function to get points
        XHat = eval_fd( XFd, self.TSpan.Target );
    end

    if args.smooth && args.points

        if self.UsesFdCoefficients
            XHatSmth = XHat;        
            XHatReg = eval_fd( XFd, self.TSpan.Regular );

        else           
            XHatSmth = smoothSeries( XHat, ...
                                     self.TSpan.Target, ...
                                     self.TSpan.Target, ...
                                     self.FDA.FdParamsTarget );
            XHatReg = smoothSeries( XHat, ...
                                    self.TSpan.Target, ...
                                    self.TSpan.Regular, ...
                                    self.FDA.FdParamsTarget );
        end

    else
        XHatSmth = [];
        XHatReg = [];

    end

end