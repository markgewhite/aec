function [ XHat, XHatSmth, XComp ] = reconstruct( self, Z, args )
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

    [ dlXHat, dlXComp{1:self.ZDimAux}] = self.predictDecoder( dlZ );

    if args.centre && self.HasCentredDecoder
        dlXHat = dlXHat + self.MeanCurveTarget;
        for i = 1:self.ZDimAux
            dlXComp{i} = dlXComp{i} + self.MeanCurveTarget;
        end
    end

    if args.points
        % convert from dlarray
        XHat = double(extractdata(dlXHat));
        XHat = squeeze(permute( XHat, [1 3 2] ));

        if args.smooth
    
            XHatSmth = smoothSeries( XHat, ...
                                     self.TSpan.Target, ...
                                     self.TSpan.Input, ...
                                     self.FDA.FdParamsInput );
        end

    else
        XHat = dlXHat;
    end

end