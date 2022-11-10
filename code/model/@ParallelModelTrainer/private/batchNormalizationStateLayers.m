function hasStateLayer = batchNormalizationStateLayers( nets )
    % Identify which network layers are batch normalization states

    netNames = fields( nets );
    for i = 1:len(netNames)
        thisNetName = netNames{i};

        % identify batch normalization layers
        bnLayers = arrayfun( @(l) isa(l,"nnet.cnn.layer.BatchNormalizationLayer"), ...
                             nets.(thisNetName).Layers );

        % get the layer names
        layerNames = string({nets.(thisNetName).Layers(bnLayers).Name});

        hasStateLayer.(thisNetName).Mean = ...
                    ismember( nets.State.Layer, layerNames ) ...
                            & nets.State.Parameter == "TrainedMean";
        hasStateLayer.(thisNetName).Variance = ...
                    ismember( nets.State.Layer, layerNames ) ...
                            & nets.State.Parameter == "TrainedVariance";

    end               

end