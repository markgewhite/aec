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
        dilations     % dilations
        scale         % leaky ReLu scale factor
        inputDropout  % initial dropout rate
        dropout       % dropout rate
        pooling       % pooling operator
        useSkips      % whether to include short circuit paths

    end

    methods

        function self = tcnModel( XDim, XOutputDim, XChannels, CDim, ...
                                   lossFcns, superArgs, args )
            % Initialize the model
            arguments
                XDim            double {mustBeInteger, mustBePositive}
                XOutputDim      double {mustBeInteger, mustBePositive}
                XChannels       double {mustBeInteger, mustBePositive}
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
            self = self@autoencoderModel( XDim, ...
                                          XOutputDim, ...
                                          XChannels, ...
                                          CDim, ...
                                          lossFcns{:}, ...
                                          superArgsCell{:}, ...
                                          hasSeqInput = true );


            % store this class's properties
            self.nHidden = args.nHidden;
            self.filterSize = args.filterSize;

            self.scale = args.scale;
            self.inputDropout = args.inputDropout;
            self.dropout = args.dropout;
            self.pooling = args.pooling;
            self.useSkips = args.useSkips;

            if self.useSkips
                % must use the same number of filters in each block
                self.nFilters = args.nFilters*ones( self.nHidden, 1 );
            else
                % progressively double filters
                self.nFilters = args.nFilters*2.^(0:self.nHidden-1);
            end
            self.dilations = 2.^((0:self.nHidden-1)*args.dilationFactor);

            % initialize the networks
            self = initEncoder( self );
            self = initDecoder( self );

        end


        function self = initEncoder( self )
            % Initialize the encoder network
            arguments
                self        tcnModel
            end

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
                [lgraphEnc, lastLayer] = addResidualBlock( ...
                    lgraphEnc, i, lastLayer, ...
                    self.filterSize, self.nFilters(i), ...
                    self.dilations(i), ...
                    self.scale, self.dropout, self.useSkips, false );
            end
            
            % add the output layers
            switch self.pooling
                case 'GlobalMax'
                    outLayers = globalMaxPooling1dLayer( 'Name', 'maxPool' );
                    poolingLayer = 'maxPool';
                case 'GlobalAvg'
                    outLayers = globalAveragePooling1dLayer( 'Name', 'avgPool' );
                    poolingLayer = 'avgPool';
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
                self        tcnModel
            end

            % define input layers
            layersDec = [ featureInputLayer( self.ZDim, 'Name', 'in' )
                          projectAndReshapeLayer( [self.XDim 1 self.XChannels ], ...
                                        self.ZDim, 'Name', 'proj' ) ];
            
            lgraphDec = layerGraph( layersDec );
            lastLayer = 'proj';
            
            for i = 1:self.nHidden
                f = self.nFilters( self.nHidden-i+1 );
                [lgraphDec, lastLayer] = addResidualBlock( ...
                    lgraphDec, i, lastLayer, ...
                    self.filterSize, f, ...
                    self.dilations(i), ...
                    self.scale, self.dropout, self.useSkips, true );
            end
            
            % add the output layers
            switch self.pooling
                case 'GlobalMax'
                    outLayers = globalMaxPooling1dLayer( 'Name', 'maxPool' );
                    poolingLayer = 'maxPool';
                case 'GlobalAvg'
                    outLayers = globalAveragePooling1dLayer( 'Name', 'avgPool' );
                    poolingLayer = 'avgPool';
            end
            
            outLayers = [ outLayers; 
                          fullyConnectedLayer( self.XOutputDim*self.XChannels, ...
                                               'Name', 'fcout' ) ];
            
            if self.XChannels > 1
                outLayers = [ outLayers; 
                          reshapeLayer( [self.XOutputDim self.XChannels], ...
                                        'Name', 'reshape' ) ];
            end
            
            lgraphDec = addLayers( lgraphDec, outLayers );
            lgraphDec = connectLayers( lgraphDec, ...
                                       lastLayer, poolingLayer );
            
            self.nets.decoder = dlnetwork( lgraphDec );

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


