function net = initDecoder( self )
    % Override the FC decoder network initialization
    arguments
        self        BranchedModel
    end

    [lgraph, inputLayerName] = initDecoderInputLayers( self );
    
    if self.HasBranchedDecoder && self.ZDimAux>1

        mask = [false( self.ZDimAux, 1 );
                true( self.ZDim - self.ZDimAux, 1 )];
        dRange = 1:self.ZDimAux;

    else
        dRange = 0;

    end

    for d = dRange

        if self.HasDecoderMasking && self.ZDimAux>1
            maskD = mask;
            maskD(d) = true;
            lgraph = addLayers( lgraph, ...
                                maskLayer( maskD, ...
                                           'ReduceDim', true, ...
                                           'Name', ['mask' num2str(100*d)] ));
            lastLayerName = ['mask' num2str(100*d)];
            lgraph = connectLayers( lgraph, ...
                                    inputLayerName, lastLayerName );
        else
            lastLayerName = inputLayerName;
        
        end

        [lgraph, lastLayerName] = self.initDecoderHiddenLayers( lgraph, lastLayerName, d*100 ); %#ok<ASGLU> 

        if self.XChannels > 1
            lgraph = [ lgraph;
                       reshapeLayer( [ self.XTargetDim self.XChannels], ...
                                      'Name', ['out' num2str(100*d)] ) ]; %#ok<AGROW> 
        end
   
    end

    net = dlnetwork( lgraph );

end
