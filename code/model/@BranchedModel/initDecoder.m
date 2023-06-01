function net = initDecoder( self )
    % Override the FC decoder network initialization
    arguments
        self        BranchedModel
    end

    [lgraph, inputLayerName] = initDecoderInputLayers( self );
    
    if self.HasBranchedDecoder && self.ZDimAux>1

        mask = [false( self.ZDimAux, 1 );
                true( self.ZDim - self.ZDimAux, 1 )];
        lgraph = addLayers( lgraph, ...
                            additionLayer( self.ZDimAux, 'Name', 'add' ) );

        dRange = 1:self.ZDimAux;

    else
        dRange = 0;

    end

    for d = dRange

        if self.HasDecoderMasking
            maskD = mask;
            maskD(d) = true;
            lgraph = addLayers( lgraph, ...
                                maskLayer( maskD, ...
                                           'ReduceDim', true, ...
                                           'Name', ['mask' num2str(100*d)] ));
            lastLayerName = ['mask' num2str(100*d)];
            lgraph = connectLayers( lgraph, ...
                                    inputLayerName, lastLayerName );
        
        end

        [lgraph, lastLayerName] = self.initDecoderHiddenLayers( lgraph, lastLayerName, d*100 );

        if self.XChannels > 1
            finalLayerName = ['reshape' num2str(100*d)];
            lgraph = [ lgraph;
                       reshapeLayer( [ self.XTargetDim self.XChannels], ...
                                      'Name', finalLayerName ) ]; %#ok<AGROW> 
        else
            finalLayerName = lastLayerName;
        end

        if self.HasBranchedDecoder
            lgraph = connectLayers( lgraph, ...
                                    finalLayerName, ...
                                    ['add/in' num2str(d)] );
        end
    
    end

    net = dlnetwork( lgraph );

end
