function net = initEncoder( self )
    % Override the FC encoder network initialization
    arguments
        self        BranchedModel
    end

    [lgraph, lastInputLayer] = self.initEncoderInputLayers;

    if self.HasBranchedEncoder

        if self.HasEncoderMasking
            lgraph = addLayers( lgraph, ...
                                concatenationLayer( 1, self.ZDim, 'Name', 'concat' ) );
        else
            lgraph = addLayers( lgraph, ...
                                additionLayer( self.ZDim, 'Name', 'add' ) );
        end
        dRange = 1:self.ZDim;

    else
        dRange = 0;

    end

    for d = dRange

        [lgraph, lastLayerName] = self.initEncoderHiddenLayers( lgraph, lastInputLayer, d*100 );
       
        if self.HasBranchedEncoder
            if self.HasEncoderMasking
                lgraph = connectLayers( lgraph, ...
                                        lastLayerName, ...
                                        ['concat/in' num2str(d)] );
            else
                lgraph = connectLayers( lgraph, ...
                                        lastLayerName, ...
                                        ['add/in' num2str(d)] );
            end
        end

    end

    net = dlnetwork( lgraph );

end