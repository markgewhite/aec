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

    [ dlXComp{1:self.ZDimAux} ] = predict( self.Nets.Decoder, dlZ );

    % sum the components to get the full reconstruction
    dlXHat = dlXComp{1};
    for i = 2:self.ZDimAux
        dlXHat = dlXHat + dlXComp{i};
    end

    % add the mean curve if centred
    if args.centre && self.HasCentredDecoder
        dlXHat = dlXHat + self.MeanCurveTarget;
    end

    if args.points
        % convert from dlarray
        XHat = double(extractdata(dlXHat));
        XHat = squeeze(permute( XHat, [1 3 2] ));

        XComp = cell( self.ZDimAux, 1 );
        for i = 1:self.ZDimAux
            XComp{i} = double(extractdata(dlXComp{i}));
            XComp{i} = squeeze(permute( XComp{i}, [1 3 2] ));
        end

        if args.smooth
    
            XHatSmth = smoothSeries( XHat, ...
                                     self.TSpan.Target, ...
                                     self.TSpan.Input, ...
                                     self.FDA.FdParamsInput );
            XCompSmth = cell( self.ZDimAux, 1 );
            for i = 1:self.ZDimAux
                XCompSmth{i} = smoothSeries( XComp{i}, ...
                                     self.TSpan.Target, ...
                                     self.TSpan.Input, ...
                                     self.FDA.FdParamsInput );
            end

        else
            XHatSmth = [];
            XCompSmth = [];

        end

    else
        XHat = dlXHat;
        XHatSmth = [];
        XComp = dlXComp;
        XCompSmth = [];

    end

end