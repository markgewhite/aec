% ************************************************************************
% Class: convModel
%
% Subclass defining a convolutional autoencoder model
%
% ************************************************************************

classdef convModel < autoencoderModel

    properties
        nHidden       % number of hidden layers
        nFilters      % number of filters aka kernels
        filterSize    % length of the filters
        stride        % filter step size
        scale         % leaky ReLu scale factor
        inputDropout  % initial dropout rate
        dropout       % dropout rate
        pooling       % pooling operator
    end

    methods

        function self = convModel( XDim, XOutputDim, XChannels, ZDim, CDim, ...
                                   lossFcns, superArgs, args )
            % Initialize the model
            arguments
                XDim            double {mustBeInteger, mustBePositive}
                XOutputDim      double {mustBeInteger, mustBePositive}
                XChannels       double {mustBeInteger, mustBePositive}
                ZDim            double {mustBeInteger, mustBePositive}
                CDim            double {mustBeInteger, mustBePositive}
            end
            arguments (Repeating)
                lossFcns     lossFunction
            end
            arguments
                superArgs.?autoencoderModel
                args.nHidden       double ...
                    {mustBeInteger, mustBePositive} = 2
                args.nFilters      double ...
                    {mustBeInteger, mustBePositive} = 32
                args.filterSize    double ...
                    {mustBeInteger, mustBePositive} = 5
                args.stride        double ...
                    {mustBeInteger, mustBePositive} = 3
                args.scale         double ...
                    {mustBeInRange(args.scale, 0, 1)} = 0.2
                args.inputDropout  double ...
                    {mustBeInRange(args.inputDropout, 0, 1)} = 0.10
                args.dropout       double ...
                    {mustBeInRange(args.dropout, 0, 1)} = 0.05
                args.pooling       char ...
                    {mustBeMember(args.pooling, ...
                      {'GlobalMax', 'GlobalAvg', 'None'} )} = 'None'
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@autoencoderModel( XDim, ...
                                          XOutputDim, ...
                                          XChannels, ...
                                          ZDim, ...
                                          CDim, ...                                          lossFcns{:}, ...
                                          lossFcns{:}, ...
                                          superArgsCell{:}, ...
                                          hasSeqInput = false );

            % store this class's properties
            self.nHidden = args.nHidden;
            self.nFilters = args.nFilters*2.^(0:self.nHidden-1);
            self.filterSize = args.filterSize;
            self.stride = args.stride;
            self.scale = args.scale;
            self.inputDropout = args.inputDropout;
            self.dropout = args.dropout;
            self.pooling = args.pooling;

            % initialize the networks
            self = initEncoder( self );
            self = initDecoder( self );

        end


        function self = initEncoder( self )
            % Initialize the encoder network
            arguments
                self        convModel
            end
            
            % define input layers
            projectionSize = [ self.XDim 1 1 ];
            layersEnc = [ featureInputLayer( self.XDim, 'Name', 'in', ...
                              'Normalization', 'zscore', ...
                              'Mean', 0, 'StandardDeviation', 1 ) 
                          dropoutLayer( self.dropout, 'Name', 'drop0' )
                          reshapeLayer( projectionSize, 'Name', 'proj' ) ];

            lgraphEnc = layerGraph( layersEnc );
            lastLayer = 'proj';
            
            % create hidden layers
            for i = 1:self.nHidden
                [lgraphEnc, lastLayer] = addBlock( lgraphEnc, i, lastLayer, ...
                    self.filterSize, self.nFilters(i), ...
                    self.scale, self.dropout, false );
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
                          fullyConnectedLayer( self.ZDim*(self.isVAE+1), ...
                                               'Name', 'out' ) ];
            
            lgraphEnc = addLayers( lgraphEnc, outLayers );
            lgraphEnc = connectLayers( lgraphEnc, ...
                                       lastLayer, poolingLayer );
        
        
            if self.isVAE
                self.nets.encoder = dlnetworkVAE( lgraphEnc, ...
                                                  nDraws = self.nVAEDraws );
            else
                self.nets.encoder = dlnetwork( lgraphEnc );
            end

        end


        function self = initDecoder( self )
            % Initialize the decoder network
            arguments
                self        convModel
            end

            % define input layers
            layersDec = [ featureInputLayer( self.ZDim, 'Name', 'in' )
                          projectAndReshapeLayer( [self.XDim 1 self.XChannels ], ...
                                        self.ZDim, 'Name', 'proj' ) ];
            
            lgraphDec = layerGraph( layersDec );
            lastLayer = 'proj';
            
            for i = 1:self.nHidden
                f = self.nFilters( self.nHidden-i+1 );
                [lgraphDec, lastLayer] = addBlock( lgraphDec, i, lastLayer, ...
                    self.filterSize, f, self.scale, self.dropout, true );
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
                          fullyConnectedLayer( self.XOutputDim*self.XChannels, 'Name', 'fcout' ) ];
            
            if self.XChannels > 1
                outLayers = [ outLayers; 
                          reshapeLayer( [self.XOutputDim self.XChannels], 'Name', 'reshape' ) ];
            end
            
            lgraphDec = addLayers( lgraphDec, outLayers );
            lgraphDec = connectLayers( lgraphDec, ...
                                       lastLayer, poolingLayer );
            
            self.nets.decoder = dlnetwork( lgraphDec );

        end


    end

end


function [ lgraph, lastLayer ] = addBlock( lgraph, i, lastLayer, ...
                        filterSize, nFilters, scale, dropout, transpose )

    % define block
    if transpose
        block = transposedConv1dLayer( filterSize, ...
                                       nFilters, ...
                                       'Cropping', 'same', ...
                                       'Name', ['conv' num2str(i)] );
    else
        block = convolution1dLayer( filterSize, ...
                                    nFilters, ...
                                    'Padding', 'same', ...
                                    'Name', ['conv' num2str(i)] );
    end
        
    block = [   block;
                layerNormalizationLayer( 'Name', ['lnorm' num2str(i)] )
                leakyReluLayer( scale, ...
                                'Name', ['relu' num2str(i)] )
                spatialDropoutLayer( dropout, ...
                                     'Name', ['drop' num2str(i)] )
                ];

    % connect layers at the front
    lgraph = addLayers( lgraph, block );
    lgraph = connectLayers( lgraph, ...
                            lastLayer, ['conv' num2str(i)] );
    
    lastLayer = ['drop' num2str(i)];

end


