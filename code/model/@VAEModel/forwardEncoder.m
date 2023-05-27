function [dlZ, state, dlMean, dlLogVars ] = forwardEncoder( self, net, dlX, superArgs )
    % Override to perform the reparameterization trick
    % returning mean and log variance for loss calculation
    arguments
        self                    VAEModel
        net                     dlnetwork
    end
    arguments (Repeating)
        dlX                     dlarray
    end
    arguments
        superArgs.?dlnetwork 
    end

    superArgsCell = namedargs2cell( superArgs );

    [ dlEncOutput, state ] = forwardEncoder@AEModel( self, net, dlX{:}, superArgsCell{:} );

    [ dlZ, dlMean, dlLogVars ] = reparameterize( dlEncOutput, self.NumEncodingDraws );

end