% ************************************************************************
% Class: fcModel
%
% Subclass defining a temporal convolutional autoencoder model
%
% ************************************************************************

classdef tcnModel < aeModel

    properties
        nHidden       % number of hidden layers
        scale         % leaky ReLu scale factor
        dropout0      % initial dropout rate
        dropout1      % dropout rate
        pooling       % pooling operator

    end

    methods

        function self = tcnModel( superArgs, args )
            % Initialize the model
            arguments
                superArgs.?aeModel
                args.nHidden    double ...
                    {mustBeInteger, mustBePositive} = 2
                args.scale      double ...
                    {mustBeInRange(args.scale, 0, 1)} = 0.2
                args.dropout0    double ...
                    {mustBeInRange(args.dropout0, 0, 1)} = 0.1
                args.dropout1    double ...
                    {mustBeInRange(args.dropout1, 0, 1)} = 0.1
                args.pooling     char ...
                    {mustBeMember(args.pooling, ...
                      {'GlobalMax', 'GlobalAvg', 'None'} )} = 'GlobalMax'
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@aeModel( superArgsCell{:} );

            % store this class's properties
            self.nHidden = args.nHidden;
            self.scale = args.scale;
            self.dropout0 = args.dropout0;
            self.dropout1 = args.dropout1;
            self.pooling = args.pooling;


            % define the encoder network
            % --------------------------
            % define input layers
            layersEnc = [ sequenceInputLayer( self.XDim, 'Name', 'in', ...
                                   'Normalization', 'zscore', ...
                                   'Mean', 0, 'StandardDeviation', 1 )
                          dropoutLayer( self.dropout0, ...
                                        'Name', 'drop0' ) ];
            lgraphEnc = layerGraph( layersEnc );
            lastLayer = 'drop0';
            
            % create hidden layers
            for i = 1:self.nHidden
                dilations = [ 2^(i*2-2) 2^(i*2-1) ];
                [lgraphEnc, lastLayer] = addResidualBlock( lgraphEnc, i, ...
                                                    dilations, lastLayer, self );
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


            % define the decoder network
            % --------------------------
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
                                                    dilations, lastLayer, self );
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
            
            if paramDec.outX(2) > 1
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
            
            
