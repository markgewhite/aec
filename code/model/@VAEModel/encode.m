function dlZ = encode( self, X, args )
    % Override encoding for VAE
    arguments
        self            VAEModel
        X               {mustBeA(X, {'ModelDataset', 'dlarray'})}
        args.convert    logical = true
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

    dlOutput = predict( self.Nets.Encoder, dlX );

    % override here
    [ dlZ, dlMean ] = reparameterize( dlOutput, 1 );

    if self.UseEncodingMean
        dlZ = dlMean;
    end
    % end of override

    if args.auxiliary
        dlZ = dlZ( 1:self.ZDimAux, : );
    end
    
    if args.convert
        dlZ = double(extractdata(gather(dlZ)))';
    end

end