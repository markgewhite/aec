function net = initEncoder( self )
    % Override the FC encoder network initialization
    arguments
        self        BranchedModel
    end

    [lgraph, lastInputLayer] = self.initEncoderInputLayers;

    if self.HasBranchedEncoder
        lgraph = addLayers( lgraph, ...
                            additionLayer( self.ZDimAux, 'Name', 'add' ) );
        if self.HasEncoderMasking
            mask = [false( self.ZDimAux, 1 );
                    true( self.ZDim - self.ZDimAux, 1 )];
        end
        dRange = 1:self.ZDimAux;

    else
        dRange = 0;

    end

    for d = dRange

        [lgraph, lastLayerName] = self.initEncoderHiddenLayers( lgraph, lastInputLayer, d*100 );
       
        if self.HasBranchedEncoder && self.HasEncoderMasking
            maskD = mask;
            maskD(d) = true;
            finalLayerName = ['mask' num2str(100*d)];
            lgraph = addLayers( lgraph, ...
                                maskLayer( maskD, ...
                                           'ReduceDim', false, ...
                                            'Name', finalLayerName ) );
        else
            finalLayerName = lastLayerName;
    
        end
    
        if self.HasBranchedEncoder
            lgraph = connectLayers( lgraph, ...
                                    finalLayerName, ...
                                    ['add/in' num2str(d)] );
        end

    end

    net = dlnetwork( lgraph );

end