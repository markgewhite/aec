function [ lgraph, lastLayer ] = addBlock( firstLayer, ...
                                lgraph, i, precedingLayer,  ...
                                scale, dropout, normType, actType )
    
    % initialize the block
    block = firstLayer;

    % add the specified type of normalization
    switch normType
        case 'Batch'
            block = [ block;
                      batchNormalizationLayer( 'Name', ...
                                    ['bnorm' num2str(i)] ) ];
        case 'Layer'
            block = [ block;
                      layerNormalizationLayer( 'Name', ...
                                    ['lnorm' num2str(i)] ) ];
    end

    % add the nonlinearity
    switch actType
        case 'Tanh'
            block = [ block;
                      tanhLayer( 'Name', ['tanh' num2str(i)] ) ];
        case 'Relu'
            if scale < 1
                block = [ block;
                          leakyReluLayer( scale, 'Name', ['relu' num2str(i)] ) ];
            end
    end

    % add dropout
    if dropout > 0
        block = [ block;
              dropoutLayer( dropout, 'Name', ['drop' num2str(i)] ) ];
    end

    % connect layers at the front
    lgraph = addLayers( lgraph, block );
    lgraph = connectLayers( lgraph, precedingLayer, firstLayer.Name );

    lastLayer = block(end).Name;

end
