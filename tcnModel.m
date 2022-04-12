% ************************************************************************
% Class: tcnModel
%
% Subclass defining a temporal convolutional autoencoder model
%
% ************************************************************************

classdef tcnModel < autoencoderModel

    properties
        nHidden       % number of hidden layers
        nFilters      % number of filters aka kernels
        filterSize    % length of the filters
        scale         % leaky ReLu scale factor
        inputDropout  % initial dropout rate
        dropout       % dropout rate
        pooling       % pooling operator
        useSkips      % whether to include short circuit paths

    end

    methods

        function self = tcnModel( lossFcns, superArgs, args )
            % Initialize the model
            arguments (Repeating)
                lossFcns     lossFunction
            end
            arguments
                superArgs.?autoencoderModel
                args.nHidden       double ...
                    {mustBeInteger, mustBePositive} = 2
                args.nFilters      double ...
                    {mustBeInteger, mustBePositive} = 16
                args.filterSize    double ...
                    {mustBeInteger, mustBePositive} = 5
                args.scale         double ...
                    {mustBeInRange(args.scale, 0, 1)} = 0.2
                args.inputDropout  double ...
                    {mustBeInRange(args.inputDropout, 0, 1)} = 0.1
                args.dropout       double ...
                    {mustBeInRange(args.dropout, 0, 1)} = 0.1
                args.pooling       char ...
                    {mustBeMember(args.pooling, ...
                      {'GlobalMax', 'GlobalAvg', 'None'} )} = 'GlobalMax'
                args.useSkips      logical = true
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@autoencoderModel( lossFcns{:}, superArgsCell{:} );

            % store this class's properties
            self.nHidden = args.nHidden;
            self.scale = args.scale;
            self.inputDropout = args.inputDropout;
            self.dropout = args.dropout;
            self.pooling = args.pooling;

            % initialize the networks
            self = initEncoder( self, args );
            self = initDecoder( self, args );

        end


        function self = initEncoder( self, args )
            % Initialize the encoder network
            
            % define input layers
            layersEnc = [ sequenceInputLayer( self.XChannels, 'Name', 'in', ...
                                   'Normalization', 'zscore', ...
                                   'Mean', 0, 'StandardDeviation', 1 )
                          dropoutLayer( self.inputDropout, ...
                                        'Name', 'drop0' ) ];
            lgraphEnc = layerGraph( layersEnc );
            lastLayer = 'drop0';
            
            % create hidden layers
            for i = 1:self.nHidden
                dilations = [ 2^(i*2-2) 2^(i*2-1) ];
                [lgraphEnc, lastLayer] = addResidualBlock( lgraphEnc, i, ...
                                                    dilations, lastLayer, args );
            end
            
            % add the output layers
            switch self.pooling
                case 'GlobalMax'
                    outLayers = globalMaxPooling1dLayer( 'Name', 'maxPool' );
                    poolingLayer = 'maxPool';
                case 'GlobalAvg'
                    outLayers = globalAveragePooling1dLayer( 'Name', 'avgPool' );
                    poolingLayer = 'avgPool';
                case 'None'
                    outLayers = [];
                    poolingLayer = 'out';
            end
            
            outLayers = [ outLayers;
                          fullyConnectedLayer( self.ZDim, 'Name', 'out' ) ];
            
            lgraphEnc = addLayers( lgraphEnc, outLayers );
            lgraphEnc = connectLayers( lgraphEnc, ...
                                       lastLayer, poolingLayer );
        
        
            self.nets.encoder = dlnetwork( lgraphEnc );
            
        end


        function self = initDecoder( self, args )
            % Initialize the decoder network

            % define input layers
            layersDec = [ featureInputLayer( self.ZDim, 'Name', 'in' )
                          projectAndReshapeLayer( [self.XDim 1 self.XChannels ], ...
                                        self.ZDim, 'Name', 'proj' ) ];
            
            lgraphDec = layerGraph( layersDec );
            lastLayer = 'proj';
            
            for i = 1:self.nHidden
                dilations = [ 2^(2*(self.nHidden-i)+1) ...
                                2^(2*(self.nHidden-i)) ];
                [lgraphDec, lastLayer] = addResidualBlock( lgraphDec, i, ...
                                                    dilations, lastLayer, args );
            end
            
            % add the output layers
            switch self.pooling
                case 'GlobalMax'
                    outLayers = globalMaxPooling1dLayer( 'Name', 'maxPool' );
                    poolingLayer = 'maxPool';
                case 'GlobalAvg'
                    outLayers = globalAveragePooling1dLayer( 'Name', 'avgPool' );
                    poolingLayer = 'avgPool';
                case 'None'
                    outLayers = [];
                    poolingLayer = 'fcout';
            end
            
            outLayers = [ outLayers; 
                          fullyConnectedLayer( self.XDim*self.XChannels, 'Name', 'fcout' ) ];
            
            if self.XChannels > 1
                outLayers = [ outLayers; 
                          reshapeLayer( [self.XDim self.XChannels], 'Name', 'reshape' ) ];
            end
            
            lgraphDec = addLayers( lgraphDec, outLayers );
            lgraphDec = connectLayers( lgraphDec, ...
                                       lastLayer, poolingLayer );
            
            self.nets.decoder = dlnetwork( lgraphDec );

        end


    end

end


function [ lgraph, lastLayer ] = addResidualBlock( ...
                                  lgraph, i, dilations, lastLayer, params )

    i1 = i*2-1;
    i2 = i1+1;

    % define residual block
    block = [   convolution1dLayer( params.filterSize, ...
                                    params.nFilters, ...
                                    'DilationFactor', dilations(1), ...
                                    'Padding', 'causal', ...
                                    'Name', ['conv' num2str(i1)] )
                layerNormalizationLayer( 'Name', ['lnorm' num2str(i1)] )
                leakyReluLayer( params.scale, ...
                                'Name', ['relu' num2str(i1)] )
                spatialDropoutLayer( params.dropout, ...
                                     'Name', ['drop' num2str(i1)] )
                convolution1dLayer( params.filterSize, ...
                                    params.nFilters, ...
                                    'DilationFactor', dilations(2), ...
                                    'Padding', 'causal', ...
                                    'Name', ['conv' num2str(i2)] )
                layerNormalizationLayer( 'Name', ['lnorm' num2str(i2)] )
                ];

    if params.useSkips
        block = [ block; 
                  additionLayer( 2, 'Name', ['add' num2str(i)] ) ];
    end

    block = [ block;
                leakyReluLayer( params.scale, ...
                                'Name', ['relu' num2str(i2)] )
                spatialDropoutLayer( params.dropout, ...
                                     'Name', ['drop' num2str(i2)] )
                ];

    % connect layers at the front
    lgraph = addLayers( lgraph, block );
    lgraph = connectLayers( lgraph, ...
                            lastLayer, ['conv' num2str(i1)] );
    
    if params.useSkips
        % include a short circuit ('skip')

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

    end

    lastLayer = ['drop' num2str(i2)];

end


