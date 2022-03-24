% ************************************************************************
% Function: aeTCNDesign
%
% Initialise autoencoder networks with a Temporal Convolution design
%
% Parameters:
%           setup       : network design parameters
%           
% Outputs:
%           dlnetEnc    : initialised encoder network
%           dlnetDec    : initialised decoder network
%
% ************************************************************************


function [ dlnetEnc, dlnetDec ] = aeTCNDesign( paramEnc, paramDec )


% define the encoder network
% --------------------------

% define input layers
layersEnc = [ featureInputLayer( paramEnc.input, 'Name', 'in', ...
                       'Normalization', 'zscore', ...
                       'Mean', 0, 'StandardDeviation', 1 )
              dropoutLayer( paramEnc.initialDropout, ...
                            'Name', 'drop0' )
              reshapeLayer( paramEnc.projectionSize, ...
                            'Name', 'proj' ) ];
lgraphEnc = layerGraph( layersEnc );
lastLayer = 'proj';

% create hidden layers
for i = 1:paramEnc.nHidden
    nDilation = 2^(i-1);
    [lgraphEnc, lastLayer] = addResidualBlock( lgraphEnc, i, ...
                                        nDilation, lastLayer, paramEnc );
end

% add the output layers
outLayers = fullyConnectedLayer( paramEnc.outZ, 'Name', 'out' );

lgraphEnc = addLayers( lgraphEnc, outLayers );
lgraphEnc = connectLayers( lgraphEnc, ...
                           lastLayer, 'out' );

dlnetEnc = dlnetwork( lgraphEnc );



% define the decoder network
% --------------------------

layersDec = [ featureInputLayer( paramDec.input, 'Name', 'in' )
              dropoutLayer( paramDec.initialDropout, ...
                            'Name', 'drop0' )
              projectAndReshapeLayer( paramDec.projectionSize, ...
                            paramDec.input, 'Name', 'proj' ) ];

lgraphDec = layerGraph( layersDec );
lastLayer = 'proj';

for i = 1:paramDec.nHidden
    nDilation = 2^(paramDec.nHidden-i);
    [lgraphDec, lastLayer] = addResidualBlock( lgraphDec, i, ...
                                        nDilation, lastLayer, paramDec );
end

% add the output layers
outLayers = [ fullyConnectedLayer( prod(paramDec.outX), 'Name', 'fcout' )
              reshapeLayer( paramDec.outX, 'Name', 'out' ) ];

lgraphDec = addLayers( lgraphDec, outLayers );
lgraphDec = connectLayers( lgraphDec, ...
                           lastLayer, 'fcout' );

dlnetDec = dlnetwork( lgraphDec );


end



function [ lgraph, lastLayer ] = addResidualBlock( ...
                                  lgraph, i, nDilation, lastLayer, params )

    % define residual block
    block = [   convolution1dLayer( params.filterSize, ...
                                    params.nFilters, ...
                                    'DilationFactor', nDilation, ...
                                    'Padding', 'causal', ...
                                    'Name', ['conv' num2str(i)] )
                batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                leakyReluLayer( params.scale, ...
                                'Name', ['relu' num2str(i)] )
                spatialDropoutLayer( params.dropout, ...
                                     'Name', ['drop' num2str(i)] )
                ];

    if params.useSkips
        % append an addition layer to the block
        block = [ block; 
                  additionLayer( 2, 'Name', ['add' num2str(i)] ) ];

        lgraph = addLayers( lgraph, block );
        lgraph = connectLayers( lgraph, ...
                            lastLayer, ['conv' num2str(i)] );

        if i == 1
            % include convolution in first skip connection
            skipLayer = convolution1dLayer( 1, params.nFilters, ...
                                            'Name', 'convSkip' );
            lgraph = addLayers( lgraph, skipLayer );
            lgraph = connectLayers( lgraph, ...
                                       lastLayer, 'convSkip' );
            lgraph = connectLayers( lgraph, ...
                               'convSkip', ['add' num2str(i) '/in2'] );
        else
            % connect the skip
            lgraph = connectLayers( lgraph, ...
                               lastLayer, ['add' num2str(i) '/in2'] );
        end
    
        lastLayer = ['add' num2str(i)];

    else
        % connect layers at the front
        lgraph = addLayers( lgraph, block );
        lgraph = connectLayers( lgraph, ...
                                lastLayer, ['conv' num2str(i)] );
        lastLayer = ['drop' num2str(i)];

    end



end