function dlZ = encode( self, X, args )
    % Encode features Z from X using the model
    arguments
        self            AEModel
        X               {mustBeA(X, {'ModelDataset', 'dlarray'})}
        args.convert     logical = true
        args.auxiliary  logical = false
    end

    if isa( X, 'ModelDataset' )
        dlX = self.getDLArrays( X );
    else
        dlX = X;
    end

    if self.FlattenInput && size( dlX, 3 ) > 1
        dlX = flattenDLArray( dlX );
    end

    dlZ = predict( self.Nets.Encoder, dlX );

    if args.auxiliary
        dlZ = dlZ( 1:self.ZDimAux, : );
    end
    
    if args.convert
        dlZ = double(extractdata(gather(dlZ)))';
    end

end