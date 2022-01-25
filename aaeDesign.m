% ************************************************************************
% Function: aaeDesign
%
% Initialise the adversarial autoencoder networks
%
% Parameters:
%           setup       : network design parameters
%           
% Outputs:
%           dlnetEnc    : initialised encoder network
%           dlnetDec    : initialised decoder network
%           dlnetDis    : initialised discriminator network
%           dlnetCls    : initialised classifier network
%
% ************************************************************************

function [ dlnetEnc, dlnetDec, dlnetDis, dlnetCls ] = ...
                aaeDesign( paramEnc, paramDec, paramDis, paramCls )


% define the encoder network
% --------------------------

% create the input layer
layersEnc = [
    featureInputLayer( paramEnc.input, 'Name', 'in', ...
                               'Normalization', 'zscore', ...
                               'Mean', 0, 'StandardDeviation', 1 )
    dropoutLayer( paramEnc.dropout, 'Name', 'drop' )
    ];
lgraphEnc = layerGraph( layersEnc );

% create the hidden layers
switch paramEnc.type

    case 'FullyConnected'
        layersEnc = [];
        for i = 1:paramEnc.nHidden
            layersEnc = [ layersEnc; ...
                fullyConnectedLayer( paramEnc.nFC, 'Name', ['fc' num2str(i)] )
                sigmoidLayer( 'Name', ['sig' num2str(i)] )
                dropoutLayer( paramEnc.dropout, 'Name', ...
                                                 ['drop' num2str(i)] )
                ]; %#ok<*AGROW> 
        end
        lgraphEnc = addLayers( lgraphEnc, layersEnc );
        lgraphEnc = connectLayers( lgraphEnc, 'drop', 'fc1' );
        lastLayer = ['drop' num2str(i)];

    case 'Convolutional'
        projectionSize = [ paramEnc.projectionSize 1 1 ];
        layersEnc = reshapeLayer( projectionSize, 'Name', 'proj' );
        filterSize = [ paramEnc.filterSize 1 ];
        stride = [ paramEnc.stride 1 ];
        for i = 1:paramEnc.nHidden
            nFilters = 2^(paramEnc.nHidden-i);
            layersEnc = [ layersEnc; ...
                convolution2dLayer( filterSize, ...
                        nFilters*paramEnc.nFilters, ...
                        'Stride', stride, ...
                        'Name', ['tconv' num2str(i)] )
                batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                leakyReluLayer( paramEnc.scale, ...
                                'Name', ['relu' num2str(i)] )
                ]; %#ok<*AGROW> 
            if paramEnc.maxPooling
                poolSize = [ paramEnc.poolSize 1 ];
                layersEnc = [ layersEnc; ...
                    maxPooling2dLayer( poolSize, ...
                            'Name', ['mpool' num2str(i)] ) ];
            end
        end
        lgraphEnc = addLayers( lgraphEnc, layersEnc );
        lgraphEnc = connectLayers( lgraphEnc, 'drop', 'proj' );
        if paramEnc.maxPooling
            lastLayer = ['mpool' num2str(i)];
        else
            lastLayer = ['relu' num2str(i)];
        end

    otherwise
        error('Unrecognised encoder network type.');

end
% create the output layers
layersEnc = fullyConnectedLayer( paramEnc.outZ, 'Name', 'out' );
lgraphEnc = addLayers( lgraphEnc, layersEnc );
lgraphEnc = connectLayers( lgraphEnc, lastLayer, 'out' );

dlnetEnc = dlnetwork( lgraphEnc );


% define the decoder network
% --------------------------

% create the input layer
layersDec = featureInputLayer( paramDec.input, 'Name', 'in' );
lgraphDec = layerGraph( layersDec );

% create the hidden layers
switch paramDec.type

    case 'FullyConnected'
        for i = 1:paramDec.nHidden
            layersDec = [ layersDec; ...
                fullyConnectedLayer( paramDec.nFC, 'Name', ['fc' num2str(i)] )
                sigmoidLayer( 'Name', ['sig' num2str(i)] )           
                ]; %#ok<*AGROW> 
        end
        lgraphDec = addLayers( lgraphDec, layersDec );
        lgraphDec = connectLayers( lgraphDec, 'in', 'fc1' );
        lastLayer = ['sig' num2str(i)];

    case 'Convolutional'
        projectionSize = [ paramDec.projectionSize 1 1 ];
        layersDec = projectAndReshapeLayer( projectionSize, ...
                                    paramDec.input, 'Name', 'proj' );
        filterSize = [ paramDec.filterSize 1 ];
        stride = [ paramDec.stride 1 ];
        for i = 1:paramDec.nHidden
            nFilters = 2^(i-1);
            layersDec = [ layersDec; ...
                transposedConv2dLayer( filterSize, ...
                        nFilters*paramDec.nFilters, ...
                        'Stride', stride, ...
                        'Name', ['tconv' num2str(i)] )
                batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                leakyReluLayer( paramDec.scale, ...
                                'Name', ['relu' num2str(i)] )            
                ]; %#ok<*AGROW> 
            if paramDec.maxPooling
                poolSize = [ paramDec.poolSize 1 ];
                layersDec = [ layersDec; ...
                    maxPooling2dLayer( poolSize, ...
                            'Name', ['mpool' num2str(i)] ) ];
            end
        end
        lgraphDec = addLayers( lgraphDec, layersDec );
        lgraphDec = connectLayers( lgraphDec, 'in', 'proj' );
        if paramDec.maxPooling
            lastLayer = ['mpool' num2str(i)];
        else
            lastLayer = ['relu' num2str(i)];
        end

    otherwise
            error('Unrecognised decoder network type.');
        
end

% create the output layer
layersDec = [
                dropoutLayer( paramDec.dropout, 'Name', 'drop' )
                fullyConnectedLayer( paramDec.outX, 'Name', 'out' )
            ];
lgraphDec = addLayers( lgraphDec, layersDec );
lgraphDec = connectLayers( lgraphDec, lastLayer, 'drop' );
dlnetDec = dlnetwork( lgraphDec );


% define the discriminator network
% -----------------------------

% create the input layer
layersDis = featureInputLayer( paramDis.input, 'Name', 'in' );

% create the hidden layers
for i = 1:paramDis.nHidden
    layersDis = [ layersDis; ...
        fullyConnectedLayer( paramDis.nFC, 'Name', ['fc' num2str(i)] )
        sigmoidLayer( 'Name', ['sig' num2str(i)] )
        ];
end

% create final layers
layersDis = [ layersDis; ...    
        dropoutLayer( paramDis.dropout, 'Name', 'drop1' )
        fullyConnectedLayer( 1, 'Name', 'fcout' )
        sigmoidLayer( 'Name', 'out' )
        ];

lgraphDis = layerGraph( layersDis );
dlnetDis = dlnetwork( lgraphDis );




% define the classifier network
% -----------------------------

% create the input layer
layersCls = featureInputLayer( paramCls.input, 'Name', 'in' );

% create the hidden layers
for i = 1:paramCls.nHidden
    layersCls = [ layersCls; ...
        fullyConnectedLayer( paramCls.nFC, 'Name', ['fc' num2str(i)] )
        sigmoidLayer( 'Name', ['sig' num2str(i)] )
        ];
end

% create final layers
layersCls = [ layersCls; ...    
        dropoutLayer( paramCls.dropout, 'Name', 'drop1' )
        fullyConnectedLayer( paramCls.output, 'Name', 'fcout' )
        sigmoidLayer( 'Name', 'out' )
        ];

lgraphCls = layerGraph( layersCls );
dlnetCls = dlnetwork( lgraphCls );


end