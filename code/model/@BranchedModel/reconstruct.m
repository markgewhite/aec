function [ XHat, XHatSmth, XComp, XCompSmth ] = reconstruct( self, Z, args )
    % Reconstruct X from Z using the model and generate components
    arguments
        self            BranchedModel
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

    [ dlXHat, dlXComp{1:self.ZDimAux}] = predict( self.Nets.Decoder, dlZ );

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
        for i = 1:self.ZDimAux
            dlXComp{i} = double(extractdata(dlXComp{i}));
            dlXComp{i} = squeeze(permute( dlXComp{i}, [1 3 2] ));
        end

        if args.smooth
    
            XHatSmth = smoothSeries( XHat, ...
                                     self.TSpan.Target, ...
                                     self.TSpan.Input, ...
                                     self.FDA.FdParamsInput );
            XCompSmth = cell( self.ZDimAux, 1 );
            for i = 1:ZDimAux
                XCompSmth{i} = smoothSeries( dlXComp{i}, ...
                                     self.TSpan.Target, ...
                                     self.TSpan.Input, ...
                                     self.FDA.FdParamsInput );
            end

        end

    else

        XHat = dlXHat;
        XHatSmth = [];
        XComp = dlXComp;
        XCompSmth = [];

    end

end