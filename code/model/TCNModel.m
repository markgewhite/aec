classdef TCNModel < FullAEModel
    % Subclass defining a temporal convolutional autoencoder model

    properties
        NumHidden     % number of hidden layers
        NumFilters    % number of filters aka kernels
        FilterSize    % length of the filters
        Dilations     % dilations
        ReLuScale     % leaky ReLu scale factor
        InputDropout  % initial dropout rate
        Dropout       % dropout rate
        Pooling       % pooling operator
        UseSkips      % whether to include short circuit paths

    end

    methods

        function self = TCNModel( thisDataset, ...
                                  lossFcns, ...
                                  superArgs, ...
                                  superArgs2, ...
                                  args )
            % Initialize the model
            arguments
                thisDataset     ModelDataset
            end
            arguments (Repeating)
                lossFcns        LossFunction
            end
            arguments
                superArgs.?FullAEModel
                superArgs2.name    string
                superArgs2.path    string
                args.numHidden     double ...
                    {mustBeInteger, mustBePositive} = 2
                args.numFilters    double ...
                    {mustBeInteger, mustBePositive} = 16
                args.filterSize    double ...
                    {mustBeInteger, mustBePositive} = 5
                args.dilationFactor  double ...
                    {mustBeInteger, mustBePositive} = 2
                args.scale         double ...
                    {mustBeInRange(args.scale, 0, 1)} = 0.2
                args.inputDropout  double ...
                    {mustBeInRange(args.inputDropout, 0, 1)} = 0.10
                args.dropout       double ...
                    {mustBeInRange(args.dropout, 0, 1)} = 0.05
                args.pooling       char ...
                    {mustBeMember(args.pooling, ...
                      {'GlobalMax', 'GlobalAvg'} )} = 'GlobalMax'
                args.useSkips      logical = true
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            superArgs2Cell = namedargs2cell( superArgs2 );

            self@FullAEModel( thisDataset, ...
                              lossFcns{:}, ...
                              superArgsCell{:}, ...
                              superArgs2Cell{:}, ...
                              hasSeqInput = true );


            % store this class's properties
            self.NumHidden = args.numHidden;
            self.FilterSize = args.filterSize;

            self.ReLuScale = args.scale;
            self.InputDropout = args.inputDropout;
            self.Dropout = args.dropout;
            self.Pooling = args.pooling;
            self.UseSkips = args.useSkips;

            if self.UseSkips
                % must use the same number of filters in each block
                self.NumFilters = args.numFilters*ones( self.NumHidden, 1 );
            else
                % progressively double filters
                self.NumFilters = args.numFilters*2.^(0:self.NumHidden-1);
            end
            self.Dilations = 2.^((0:self.NumHidden-1)*args.dilationFactor);

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
        
        
            if self.IsVAE
                net = VAEdlnetwork( lgraphEnc, numDraws = self.NumVAEDraws );
            else
                net = dlnetwork( lgraphEnc );
            end

        end


        function net = initDecoder( self )
            % Initialize the decoder network
            arguments
                self        TCNModel
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


