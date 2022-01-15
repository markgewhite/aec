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
switch paramEnc.type

    case 'FullyConnected'
        layersEnc = [
            featureInputLayer( paramEnc.input, 'Name', 'in', ...
                               'Normalization', 'zscore', ...
                               'Mean', 0, 'StandardDeviation', 1 )

            fullyConnectedLayer( 100, 'Name', 'fc1' )
            sigmoidLayer( 'Name', 'sig1' )
            
            dropoutLayer( paramEnc.dropout, 'Name', 'drop2' )
            fullyConnectedLayer( paramEnc.outZ, 'Name', 'fc2' )
            ];

    case 'Convolutional'
        layersEnc = [
            featureInputLayer( paramEnc.input, 'Name', 'in', ...
                               'Normalization', 'zscore', ...
                               'Mean', 0, 'StandardDeviation', 1 )
            reshapeLayer( paramEnc.projectionSize, 'Name', 'proj' )
            %projectAndReshapeLayer( paramEnc.projectionSize, ...
            %                        paramEnc.input, 'Name', 'proj' )

            convolution2dLayer( paramEnc.filterSize, ...
                                    2*paramEnc.nFilters, 'Name', 'tconv1' )
            batchNormalizationLayer( 'Name', 'bnorm1' )
            reluLayer( 'Name', 'relu1' )

            convolution2dLayer( paramEnc.filterSize, ...
                                    paramEnc.nFilters, 'Name', 'tconv2' )
            batchNormalizationLayer('Name','bnorm2')
            reluLayer( 'Name', 'relu2' )

            dropoutLayer( paramEnc.dropout, 'Name', 'drop2' )
            fullyConnectedLayer( paramEnc.outZ, 'Name', 'fc2' )
            ];

    otherwise
        error('Unrecognised encoder network type.');

end

lgraphEnc = layerGraph( layersEnc );
dlnetEnc = dlnetwork( lgraphEnc );


% define the decoder network
switch paramDec.type

    case 'FullyConnected'
        layersDec = [
            featureInputLayer( paramDec.input, 'Name', 'in' )
            fullyConnectedLayer( 100, 'Name', 'fc1' )
            sigmoidLayer( 'Name', 'sig1' )
            dropoutLayer( paramDec.dropout, 'Name', 'drop2' )
            fullyConnectedLayer( paramDec.outX, 'Name', 'fc2' )
            ];

    case 'Convolutional'
        layersDec = [
            featureInputLayer( paramDec.input, 'Name', 'in' )
            projectAndReshapeLayer( paramDec.projectionSize, ...
                                    paramDec.input, 'Name', 'proj' )
            dropoutLayer( paramDec.dropout, 'Name', 'drop1' )

            transposedConv2dLayer( paramDec.filterSize, ...
                                    paramDec.nFilters, 'Name', 'conv1' )
            batchNormalizationLayer( 'Name', 'bnorm1' )
            reluLayer( 'Name', 'relu1' )

            transposedConv2dLayer( paramDec.filterSize, ...
                                    2*paramDec.nFilters, 'Name', 'conv2' )
            batchNormalizationLayer('Name','bnorm2')
            reluLayer( 'Name', 'relu2' )

            fullyConnectedLayer( paramDec.outX, 'Name', 'fc2' )
            ];
    
    otherwise
            error('Unrecognised decoder network type.');
        
end

lgraphDec = layerGraph( layersDec );
dlnetDec = dlnetwork( lgraphDec );


% define the classifier network
layersCls = [
    featureInputLayer( paramCls.input, 'Name', 'in' )
    fullyConnectedLayer( 21, 'Name', 'fc1' )
    sigmoidLayer( 'Name', 'sig1' )
    dropoutLayer( paramCls.dropout, 'Name', 'drop1' )
    fullyConnectedLayer( paramCls.output, 'Name', 'fc2' )
    sigmoidLayer( 'Name', 'out' )
    ];

lgraphCls = layerGraph( layersCls );
dlnetCls = dlnetwork( lgraphCls );


end