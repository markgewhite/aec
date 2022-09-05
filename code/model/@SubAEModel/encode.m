function dlZ = encode( self, X, arg )
    % Encode features Z from X using the model
    arguments
        self            SubAEModel
        X               {mustBeA(X, {'ModelDataset', 'dlarray'})}
        arg.convert     logical = true
    end

    if isa( X, 'ModelDataset' )
        dlX = X.getDLInput( self.XDimLabels );
    else
        dlX = X;
    end

    if self.FlattenInput && size( dlX, 3 ) > 1
        dlX = flattenDLArray( dlX );
    end

    dlZ = predict( self.Nets.Encoder, dlX );
    
    if arg.convert
        dlZ = double(extractdata( dlZ ))';
    end

end