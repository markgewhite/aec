function [outputs, state] = forwardEncoder( self, net, dlX, superArgs )
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

    [ outputs, state ] = forwardEncoder@AEModel( self, net, dlX{:}, superArgsCell{:} );

    [ outputs.dlZ, outputs.dlMu, outputs.dlLogVar ] = ...
                reparameterize( outputs.dlZ, self.NumEncodingDraws );

end