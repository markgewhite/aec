classdef ConvolutionalModel < FCModel
    % Subclass defining a convolutional autoencoder model

    properties
        NumFilters    % number of filters aka kernels
        FilterSize    % length of the filters
        Stride        % filter step size
        Padding       % encoder padding
        Pooling       % pooling operator
        NumHiddenDecoder   % number of decoder layers
        FilterSizeDecoder  % length of the decoder filters
        StrideDecoder      % decoder stride
        PaddingDecoder  % decoder padding better known as cropping
    end

    methods

        function self = ConvolutionalModel( thisDataset, ...
                                            superArgs, ...
                                            superArgs2, ...
                                            args )
            % Initialize the model
            arguments
                thisDataset         ModelDataset
                superArgs.?FCModel
                superArgs2.name     string
                superArgs2.path     string
                args.NumFilters     double ...
                    {mustBeInteger, mustBePositive} = 16
                args.FilterSize     double ...
                    {mustBeInteger, mustBePositive} = 5
                args.Stride         double ...
                    {mustBeInteger, mustBePositive} = 3
                args.Padding        char ...
                    {mustBeMember(args.Padding, ...
                      {'Same', 'None'} )} = 'None'
                args.Pooling        char ...
                    {mustBeMember(args.Pooling, ...
                      {'GlobalMax', 'GlobalAvg', 'None'} )} = 'None'
                args.NumHiddenDecoder   double ...
                    {mustBeInteger, mustBePositive} = 2
                args.FilterSizeDecoder  double ...
                    {mustBeInteger, mustBePositive} = 5
                args.StrideDecoder  double ...
                    {mustBeInteger, mustBePositive} = 3
                args.PaddingDecoder char ...
                    {mustBeMember(args.PaddingDecoder, ...
                      {'Same', 'None'} )} = 'None'
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            superArgs2Cell = namedargs2cell( superArgs2 );

            self@FCModel( thisDataset, ...
                          superArgsCell{:}, ...
                          superArgs2Cell{:}, ...
                          FlattenInput = true, ...
                          HasSeqInput = false );

            % store this class's properties
            self.NumFilters = args.NumFilters*2.^(0:self.NumHidden-1);
            self.FilterSize = args.FilterSize;
            self.Stride = args.Stride;
            self.Padding = args.Padding;
            self.Pooling = args.Pooling;
            self.NumHiddenDecoder = args.NumHiddenDecoder;
            self.FilterSizeDecoder = args.FilterSizeDecoder;
            self.StrideDecoder = args.StrideDecoder;
            self.PaddingDecoder = args.PaddingDecoder;

        end


        function net = initEncoder( self )
            % Initialize the encoder network
            arguments
                self        ConvolutionalModel
            end
            
            % define input layers
            projectionSize = [ self.XInputDim self.XChannels 1 ];
            layersEnc = [ featureInputLayer( self.XInputDim*self.XChannels, 'Name', 'in', ...
                              'Normalization', 'zscore', ...
                              'Mean', 0, 'StandardDeviation', 1 ) 
                          dropoutLayer( self.Dropout, 'Name', 'drop0' )
                          reshapeLayer( projectionSize, 'Name', 'proj' ) ];

            lgraphEnc = layerGraph( layersEnc );
            lastLayer = 'proj';
            
            % create hidden layers
            for i = 1:self.NumHidden
                [lgraphEnc, lastLayer] = addBlock( lgraphEnc, i, lastLayer, ...
                    self.NumFilters(i), self.FilterSize, self.Stride, ...
                    self.Padding, self.ReluScale, self.Dropout, false );
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
                          fullyConnectedLayer( self.ZDim, 'Name', 'out' ) ];
            
            lgraphEnc = addLayers( lgraphEnc, outLayers );
            lgraphEnc = connectLayers( lgraphEnc, ...
                                       lastLayer, poolingLayer );
        
            net = dlnetwork( lgraphEnc );

        end


        function net = initDecoder( self )
            % Initialize the decoder network
            arguments
                self        ConvolutionalModel
            end

            % define input layers
            layersDec = [ featureInputLayer( self.ZDim, 'Name', 'in' );
                          reshapeLayer( [1 self.ZDim], 'Name', 'reshape' ) ];
            
            lgraphDec = layerGraph( layersDec );
            lastLayer = 'reshape';
            
            for i = 1:self.NumHiddenDecoder
                [lgraphDec, lastLayer] = addBlock( lgraphDec, i, lastLayer, ...
                    self.ZDim, self.FilterSizeDecoder, ...
                    self.StrideDecoder, self.PaddingDecoder, ...
                    self.ReluScale, self.Dropout, true );
            end
            
            % add the output layers
            outLayers = convolution1dLayer( 1, 1, 'Name', 'add' );
            
            if self.XChannels > 1
                outLayers = [ outLayers; 
                          reshapeLayer( [self.XTargetDim self.XChannels], 'Name', 'reshape' ) ];
            end
            
            lgraphDec = addLayers( lgraphDec, outLayers );
            lgraphDec = connectLayers( lgraphDec, ...
                                       lastLayer, 'add' );
            
            net = dlnetwork( lgraphDec );

        end


        function dlZ = predict( encoder, X, arg )
            % Override the predict function of a fully-connected network
            arguments
                encoder         dlnetwork
                X               {mustBeA( X, {'dlarray', 'ModelDataset'} )}
                arg.convert     logical = true
            end

            dlZ = predict@AEModel( encoder, X, arg );

        end

    end

end


function [ lgraph, lastLayer ] = addBlock( lgraph, i, lastLayer, ...
                    nFilters, filterSize, stride, padding, scale, dropout, transpose )

    switch lower(padding)
        case 'none'
            pad = 0;
        case 'same'
            pad = 'same';
    end
           

    % define block            
    if transpose
        block = transposedConv1dLayer( filterSize, ...
                                       nFilters, ...
                                       'Stride', stride, ...
                                       'Cropping', pad, ...
                                       'Name', ['conv' num2str(i)] );
    else
        block = convolution1dLayer( filterSize, ...
                                    nFilters, ...
                                    'Stride', stride, ...
                                    'Padding', pad, ...
                                    'Name', ['conv' num2str(i)] );
    end
        
    block = [   block;
                batchNormalizationLayer( 'Name', ['norm' num2str(i)] )
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


