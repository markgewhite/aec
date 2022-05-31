% ************************************************************************
% Class: convModel
%
% Subclass defining a convolutional autoencoder model
%
% ************************************************************************

classdef convModel < autoencoderModel

    properties
        NumHidden     % number of hidden layers
        NumFilters    % number of filters aka kernels
        FilterSize    % length of the filters
        Stride        % filter step size
        Scale         % leaky ReLu scale factor
        InputDropout  % initial dropout rate
        Dropout       % dropout rate
        Pooling       % pooling operator
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
            self.NumHidden = args.nHidden;
            self.NumFilters = args.nFilters*2.^(0:self.NumHidden-1);
            self.FilterSize = args.filterSize;
            self.Stride = args.stride;
            self.Scale = args.scale;
            self.InputDropout = args.inputDropout;
            self.Dropout = args.dropout;
            self.Pooling = args.pooling;

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
                          dropoutLayer( self.Dropout, 'Name', 'drop0' )
                          reshapeLayer( projectionSize, 'Name', 'proj' ) ];

            lgraphEnc = layerGraph( layersEnc );
            lastLayer = 'proj';
            
            % create hidden layers
            for i = 1:self.NumHidden
                [lgraphEnc, lastLayer] = addBlock( lgraphEnc, i, lastLayer, ...
                    self.FilterSize, self.NumFilters(i), ...
                    self.Scale, self.Dropout, false );
            end
            
            % add the output layers
            switch self.Pooling
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
                          fullyConnectedLayer( self.ZDim*(self.IsVAE+1), ...
                                               'Name', 'out' ) ];
            
            lgraphEnc = addLayers( lgraphEnc, outLayers );
            lgraphEnc = connectLayers( lgraphEnc, ...
                                       lastLayer, poolingLayer );
        
        
            if self.IsVAE
                self.Nets.Encoder = dlnetworkVAE( lgraphEnc, ...
                                                  nDraws = self.NumVAEDraws );
            else
                self.Nets.Encoder = dlnetwork( lgraphEnc );
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
            
            for i = 1:self.NumHidden
                f = self.NumFilters( self.NumHidden-i+1 );
                [lgraphDec, lastLayer] = addBlock( lgraphDec, i, lastLayer, ...
                    self.FilterSize, f, self.Scale, self.Dropout, true );
            end
            
            % add the output layers
            switch self.Pooling
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
            
            self.Nets.Decoder = dlnetwork( lgraphDec );

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


