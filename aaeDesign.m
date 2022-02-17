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
    dropoutLayer( paramEnc.dropout, 'Name', 'drop0' )
    ];

% create the hidden layers
switch paramEnc.type

    case 'FullyConnected'
        for i = 1:paramEnc.nHidden
            nNodes = fix( paramEnc.nFC*2^(1-i) );
            layersEnc = [ layersEnc; ...
                fullyConnectedLayer( nNodes, 'Name', ['fc' num2str(i)] )
                batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                leakyReluLayer( paramEnc.scale, ...
                                'Name', ['relu' num2str(i)] )
                dropoutLayer( paramEnc.dropout, 'Name', ...
                                                 ['drop' num2str(i)] )
                ]; %#ok<*AGROW> 
        end

    case 'Convolutional'
        projectionSize = [ paramEnc.projectionSize 1 1 ];
        layersEnc = [ layersEnc; ...
                      reshapeLayer( projectionSize, 'Name', 'proj' ) ];
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
        end

    otherwise
        error('Unrecognised encoder network type.');

end
% create the output layers
layersEnc = [ layersEnc; ...
              fullyConnectedLayer( paramEnc.outZ, 'Name', 'out' ) ];
lgraphEnc = layerGraph( layersEnc );
dlnetEnc = dlnetwork( lgraphEnc );


% define the decoder network
% --------------------------

% create the input layer
layersDec = featureInputLayer( paramDec.input, 'Name', 'in' );

% create the hidden layers
switch paramDec.type

    case 'FullyConnected'
        for i = 1:paramDec.nHidden
            nNodes = fix( paramDec.nFC*2^(-paramDec.nHidden+i) );
            layersDec = [ layersDec; ...
                fullyConnectedLayer( nNodes, 'Name', ['fc' num2str(i)] )
                batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                leakyReluLayer( paramDec.scale, ...
                                'Name', ['relu' num2str(i)] )           
                ]; %#ok<*AGROW> 
        end

    case 'Convolutional'
        projectionSize = [ paramDec.projectionSize 1 1 ];
        layersDec = [ layersDec; ...
                      projectAndReshapeLayer( projectionSize, ...
                                    paramDec.input, 'Name', 'proj' ) ];
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
        end

    otherwise
            error('Unrecognised decoder network type.');
        
end

% create the output layer
layersDec = [ layersDec; ...
              fullyConnectedLayer( paramDec.outX, 'Name', 'out' ) ];
lgraphDec = layerGraph( layersDec );
dlnetDec = dlnetwork( lgraphDec );


% define the discriminator network
% -----------------------------

% create the input layer
layersDis = featureInputLayer( paramDis.input, 'Name', 'in' );

% create the hidden layers
for i = 1:paramDis.nHidden
    layersDis = [ layersDis; ...
        fullyConnectedLayer( paramDis.nFC, 'Name', ['fc' num2str(i)] )
        leakyReluLayer( paramDis.scale, 'Name', ['relu' num2str(i)] )
        dropoutLayer( paramDis.dropout, 'Name', ['drop' num2str(i)] )
        ];
end

% create final layers
layersDis = [ layersDis; ...    
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
        leakyReluLayer( paramCls.scale, 'Name', ['relu' num2str(i)] )
        dropoutLayer( paramCls.dropout, 'Name', ['drop' num2str(i)] )
        ];
end

% create final layers
layersCls = [ layersCls; ...    
        fullyConnectedLayer( paramCls.output, 'Name', 'fcout' )
        sigmoidLayer( 'Name', 'out' )
        ];

lgraphCls = layerGraph( layersCls );
dlnetCls = dlnetwork( lgraphCls );


end