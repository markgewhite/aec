function [ dlXHat, XHatSmth, XHatReg ] = reconstruct( self, Z, args )
    % Reconstruct X from Z using the model
    arguments
        self            AEModel
        Z               {mustBeA(Z, {'double', 'dlarray'})}
        args.centre     logical = true
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

    dlXHat = double(extractdata(gather(dlXHat)));
       
    if self.UsesFdCoefficients
        % convert to real points

        % create a dummy Fd object
        XFd = smooth_basis( self.TSpan.Target, ...
                            dlXHat(1:length(self.TSpan.Target),:,:), ...
                            self.FDA.FdParamsTarget );

        % impose the coefficient matrix
        XFd = putcoef( XFd, dlXHat );

        % evaluate the function to get points
        dlXHat = eval_fd( XFd, self.TSpan.Target );
    end

    if args.smooth

        if self.UsesFdCoefficients
            XHatSmth = dlXHat;        
            XHatReg = eval_fd( XFd, self.TSpan.Regular );

        else
            dlXHat = permute( dlXHat, [1 3 2] );
            dlXHat = squeeze( dlXHat ); 
            
            XHatSmth = smoothSeries( dlXHat, ...
                                     self.TSpan.Target, ...
                                     self.TSpan.Target, ...
                                     self.FDA.FdParamsTarget );
            XHatReg = smoothSeries( dlXHat, ...
                                    self.TSpan.Target, ...
                                    self.TSpan.Regular, ...
                                    self.FDA.FdParamsTarget );
        end

    else
        XHatSmth = [];
        XHatReg = [];

    end

end