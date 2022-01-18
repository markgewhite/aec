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
%           dlnetCls    : initialised classifier network
%
% ************************************************************************

function [ dlnetEnc, dlnetDec, dlnetCls ] = ...
                aaeDesign( paramEnc, paramDec, paramCls )


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
                ]; %#ok<*AGROW> 
        end
        lgraphEnc = addLayers( lgraphEnc, layersEnc );
        lgraphEnc = connectLayers( lgraphEnc, 'drop', 'fc1' );
        lastLayer = ['sig' num2str(i)];

    case 'Convolutional'
        layersEnc = reshapeLayer( paramEnc.projectionSize, 'Name', 'proj' );
        for i = 1:paramEnc.nHidden
            nFilters = 2^(paramEnc.nHidden-i);
            layersEnc = [ layersEnc; ...
                convolution2dLayer( paramEnc.filterSize, ...
                        nFilters*paramEnc.nFilters, ...
                        'Stride', paramEnc.stride, ...
                        'Name', ['tconv' num2str(i)] )
                batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                reluLayer( 'Name', ['relu' num2str(i)] )         
                ]; %#ok<*AGROW> 
        end
        lgraphEnc = addLayers( lgraphEnc, layersEnc );
        lgraphEnc = connectLayers( lgraphEnc, 'drop', 'proj' );
        lastLayer = ['relu' num2str(i)];

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
        layersDec = projectAndReshapeLayer( paramDec.projectionSize, ...
                                    paramDec.input, 'Name', 'proj' );
        for i = 1:paramDec.nHidden
            nFilters = 2^(i-1);
            layersDec = [ layersDec; ...
                transposedConv2dLayer( paramDec.filterSize, ...
                        nFilters*paramDec.nFilters, ...
                        'Stride', paramDec.stride, ...
                        'Name', ['tconv' num2str(i)] )
                batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                reluLayer( 'Name', ['relu' num2str(i)] )            
                ]; %#ok<*AGROW> 
        end
        lgraphDec = addLayers( lgraphDec, layersDec );
        lgraphDec = connectLayers( lgraphDec, 'in', 'proj' );
        lastLayer = ['relu' num2str(i)];
    
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