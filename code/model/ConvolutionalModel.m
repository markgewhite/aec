classdef ConvolutionalModel < FullAEModel
    % Subclass defining a convolutional autoencoder model

    properties
        NumHidden     % number of hidden layers
        NumFilters    % number of filters aka kernels
        FilterSize    % length of the filters
        Stride        % filter step size
        ReLuScale     % leaky ReLu scale factor
        InputDropout  % initial dropout rate
        Dropout       % dropout rate
        Pooling       % pooling operator
    end

    methods

        function self = ConvolutionalModel( thisDataset, ...
                                            lossFcns, ...
                                            superArgs, ...
                                            args )
            % Initialize the model
            arguments
                thisDataset         ModelDataset
            end
            arguments (Repeating)
                lossFcns            LossFunction
            end
            arguments
                superArgs.?FullAEModel
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
            self@FullAEModel( thisDataset, ...
                              lossFcns{:}, ...
                              superArgsCell{:}, ...
                              flattenInput = true, ...
                              hasSeqInput = false );

            % store this class's properties
            self.NumHidden = args.nHidden;
            self.NumFilters = args.nFilters*2.^(0:self.NumHidden-1);
            self.FilterSize = args.filterSize;
            self.Stride = args.stride;
            self.ReLuScale = args.scale;
            self.InputDropout = args.inputDropout;
            self.Dropout = args.dropout;
            self.Pooling = args.pooling;

        end


        function net = initEncoder( self )
            % Initialize the encoder network
            arguments
                self        ConvolutionalModel
            end
            
            % define input layers
            projectionSize = [ self.XInputDim 1 1 ];
            layersEnc = [ featureInputLayer( self.XInputDim, 'Name', 'in', ...
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
                    self.ReLuScale, self.Dropout, false );
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
                net = VAEdlnetwork( lgraphEnc, numDraws = self.NumVAEDraws );
            else
                net = dlnetwork( lgraphEnc );
            end

        end


        function net = initDecoder( self )
            % Initialize the decoder network
            arguments
                self        ConvolutionalModel
            end

            % define input layers
            layersDec = [ featureInputLayer( self.ZDim, 'Name', 'in' )
                          projectAndReshapeLayer( [self.XInputDim 1 self.XChannels ], ...
                                        self.ZDim, 'Name', 'proj' ) ];
            
            lgraphDec = layerGraph( layersDec );
            lastLayer = 'proj';
            
            for i = 1:self.NumHidden
                f = self.NumFilters( self.NumHidden-i+1 );
                [lgraphDec, lastLayer] = addBlock( lgraphDec, i, lastLayer, ...
                    self.FilterSize, f, self.ReLuScale, self.Dropout, true );
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
                          fullyConnectedLayer( self.XTargetDim*self.XChannels, 'Name', 'fcout' ) ];
            
            if self.XChannels > 1
                outLayers = [ outLayers; 
                          reshapeLayer( [self.XTargetDim self.XChannels], 'Name', 'reshape' ) ];
            end
            
            lgraphDec = addLayers( lgraphDec, outLayers );
            lgraphDec = connectLayers( lgraphDec, ...
                                       lastLayer, poolingLayer );
            
            net = dlnetwork( lgraphDec );

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


