classdef TCNModel < FCModel
    % Subclass defining a temporal convolutional autoencoder model

    properties
        NumFilters    % number of filters aka kernels
        FilterSize    % length of the filters
        Dilations     % dilations
        Pooling       % pooling operator
        UseSkips      % whether to include short circuit paths
        HasFCDecoder  % whether the decoder inherits the fully-connected design
    end

    methods

        function self = TCNModel( thisDataset, ...
                                  lossFcns, ...
                                  superArgs, ...
                                  superArgs2, ...
                                  args )
            % Initialize the model
            arguments
                thisDataset         ModelDataset
            end
            arguments (Repeating)
                lossFcns            LossFunction
            end
            arguments
                superArgs.?FCModel
                superArgs2.name     string
                superArgs2.path     string
                args.NumFilters     double ...
                    {mustBeInteger, mustBePositive} = 16
                args.FilterSize     double ...
                    {mustBeInteger, mustBePositive} = 5
                args.DilationFactor double ...
                    {mustBeInteger, mustBePositive} = 2
                args.Pooling        char ...
                    {mustBeMember(args.Pooling, ...
                      {'GlobalMax', 'GlobalAvg'} )} = 'GlobalMax'
                args.UseSkips       logical = true
                args.HasFCDecoder   logical = false
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            superArgs2Cell = namedargs2cell( superArgs2 );

            self@FCModel( thisDataset, ...
                          lossFcns{:}, ...
                          superArgsCell{:}, ...
                          superArgs2Cell{:}, ...
                          FlattenInput = false, ...
                          HasSeqInput = true );


            % store this class's properties
            self.FilterSize = args.FilterSize;
            self.Pooling = args.Pooling;
            self.UseSkips = args.UseSkips;
            self.HasFCDecoder = args.HasFCDecoder;

            if self.UseSkips
                % must use the same number of filters in each block
                self.NumFilters = args.NumFilters*ones( self.NumHidden, 1 );
            else
                % progressively double filters
                self.NumFilters = args.NumFilters*2.^(0:self.NumHidden-1);
            end
            self.Dilations = 2.^((0:self.NumHidden-1)*args.DilationFactor);

        end


        function net = initEncoder( self )
            % Initialize the encoder network
            arguments
                self        TCNModel
            end

            % define input layers
            layersEnc = [ sequenceInputLayer( self.XChannels, 'Name', 'in', ...
                                   'Normalization', 'zscore', ...
                                   'Mean', 0, 'StandardDeviation', 1 )
                          dropoutLayer( self.InputDropout, ...
                                        'Name', 'drop0' ) ];
            lgraphEnc = layerGraph( layersEnc );
            lastLayer = 'drop0';
            
            % create hidden layers
            for i = 1:self.NumHidden
                [lgraphEnc, lastLayer] = addResidualBlock( ...
                    lgraphEnc, i, lastLayer, ...
                    self.FilterSize, self.NumFilters(i), ...
                    self.Dilations(i), ...
                    self.ReLuScale, self.Dropout, self.UseSkips, false );
            end
            
            % add the output layers
            switch self.Pooling
                case 'GlobalMax'
                    outLayers = globalMaxPooling1dLayer( 'Name', 'maxPool' );
                    poolingLayer = 'maxPool';
                case 'GlobalAvg'
                    outLayers = globalAveragePooling1dLayer( 'Name', 'avgPool' );
                    poolingLayer = 'avgPool';
            end
            
            outLayers = [ outLayers;
                          fullyConnectedLayer( self.ZDim*(self.IsVAE+1), ...
                                               'Name', 'out' ) ];
            
            lgraphEnc = addLayers( lgraphEnc, outLayers );
            lgraphEnc = connectLayers( lgraphEnc, ...
                                       lastLayer, poolingLayer );
        
            net = dlnetwork( lgraphEnc );

        end


        function net = initDecoder( self )
            % Initialize the decoder network
            arguments
                self        TCNModel
            end

            if self.HasFCDecoder
                net = initDecoder@FCModel( self );
                return
            end

            % define input layers
            layersDec = [ featureInputLayer( self.ZDim, 'Name', 'in' )
                          projectAndReshapeLayer( [self.XInputDim 1 self.XChannels ], ...
                                        self.ZDim, 'Name', 'proj' ) ];
            
            lgraphDec = layerGraph( layersDec );
            lastLayer = 'proj';
            
            for i = 1:self.NumHidden
                f = self.NumFilters( self.NumHidden-i+1 );
                [lgraphDec, lastLayer] = addResidualBlock( ...
                    lgraphDec, i, lastLayer, ...
                    self.FilterSize, f, ...
                    self.Dilations(i), ...
                    self.ReLuScale, self.Dropout, self.UseSkips, true );
            end
            
            % add the output layers
            switch self.Pooling
                case 'GlobalMax'
                    outLayers = globalMaxPooling1dLayer( 'Name', 'maxPool' );
                    poolingLayer = 'maxPool';
                case 'GlobalAvg'
                    outLayers = globalAveragePooling1dLayer( 'Name', 'avgPool' );
                    poolingLayer = 'avgPool';
            end
            
            outLayers = [ outLayers; 
                          fullyConnectedLayer( self.XTargetDim*self.XChannels, ...
                                               'Name', 'fcout' ) ];
            
            if self.XChannels > 1
                outLayers = [ outLayers; 
                          reshapeLayer( [self.XTargetDim self.XChannels], ...
                                        'Name', 'reshape' ) ];
            end
            
            lgraphDec = addLayers( lgraphDec, outLayers );
            lgraphDec = connectLayers( lgraphDec, ...
                                       lastLayer, poolingLayer );
            
            net = dlnetwork( lgraphDec );

        end


        function dlZ = predict( encoder, X, arg )
            % Override the predict function of a fully-connected network
            arguments
                encoder         dlnetwork
                X               {mustBeA( X, {'dlarray', 'ModelDataset'} )}
                arg.convert     logical = true
            end

            dlZ = predict@FullAEModel( encoder, X, arg );

        end

    end

end


function [ lgraph, lastLayer ] = addResidualBlock( ...
                        lgraph, i, lastLayer, ...
                        filterSize, nFilters, ...
                        dilation, ...
                        scale, dropout, skip, transpose )

    % define residual block
    if transpose
        block = transposedConv1dLayer( filterSize, ...
                                       nFilters, ...
                                       'Cropping', 'same', ...
                                       'Name', ['conv' num2str(i)] );
    else
        block = convolution1dLayer( filterSize, ...
                                    nFilters, ...
                                    'DilationFactor', dilation, ...
                                    'Padding', 'causal', ...
                                    'Name', ['conv' num2str(i)] );
    end
        
    block = [   block;
                layerNormalizationLayer( 'Name', ['lnorm' num2str(i)] )
                ];

    if skip
        block = [ block; 
                  additionLayer( 2, 'Name', ['add' num2str(i)] ) ];
    end

    block = [ block;
                leakyReluLayer( scale, ...
                                'Name', ['relu' num2str(i)] )
                spatialDropoutLayer( dropout, ...
                                     'Name', ['drop' num2str(i)] )
                ];

    % connect layers at the front
    lgraph = addLayers( lgraph, block );
    lgraph = connectLayers( lgraph, ...
                            lastLayer, ['conv' num2str(i)] );
    
    if skip
        % include a short circuit ('skip')

        if i == 1
            % include convolution in first skip connection
            skipLayer = convolution1dLayer( 1, nFilters, ...
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

    lastLayer = ['drop' num2str(i)];

end


