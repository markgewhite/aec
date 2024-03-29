classdef ConvBranchedModel < BranchedModel
    % Subclass defining a convolutional autoencoder model
    properties
        NumFilters    % number of filters aka kernels for the encoder
        FilterFactor  % multiple applied to NumFilters for each successive layer
        FilterSize    % length of the filters for the encoder
        Stride        % filter step size for the encocer
        Padding       % encoder paddingfor the encoder
        Pooling       % pooling operator
        NumFiltersDecoder  % number of filters for the decoder
        FilterFactorDecoder  % multiple applied to NumFiltersDecoder for each successive layer
        FilterSizeDecoder  % length of the decoder filters
        StrideDecoder      % decoder stride
        PaddingDecoder  % decoder padding better known as cropping
    end

    methods

        function self = ConvBranchedModel( thisDataset, ...
                                           superArgs, ...
                                           superArgs2, ...
                                           args )
            % Initialize the model
            arguments
                thisDataset     ModelDataset
                superArgs.?BranchedModel
                superArgs2.name     string
                superArgs2.path     string
                args.NumFilters     double ...
                    {mustBeInteger, mustBePositive} = 16
                args.FilterFactor   double ...
                    {mustBeInteger, mustBePositive} = 2
                args.FilterSize     double ...
                    {mustBeInteger, mustBePositive} = 4
                args.Stride         double ...
                    {mustBeInteger, mustBePositive} = 3
                args.Padding        char ...
                    {mustBeMember(args.Padding, ...
                      {'Same', 'None'} )} = 'None'
                args.Pooling        char ...
                    {mustBeMember(args.Pooling, ...
                      {'GlobalMax', 'GlobalAvg', 'None'} )} = 'None'
                args.NumFiltersDecoder  double ...
                    {mustBeInteger, mustBePositive} = 16
                args.FilterFactorDecoder  double ...
                    {mustBeInteger, mustBePositive} = 2
                args.FilterSizeDecoder  double ...
                    {mustBeInteger, mustBePositive} = 4
                args.StrideDecoder  double ...
                    {mustBeInteger, mustBePositive} = 2
                args.PaddingDecoder char ...
                    {mustBeMember(args.PaddingDecoder, ...
                      {'Same', 'None'} )} = 'None'
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            superArgs2Cell = namedargs2cell( superArgs2 );

            self@BranchedModel( thisDataset, ...
                          superArgsCell{:}, ...
                          superArgs2Cell{:} );

            % store this class's properties
            self.NumFilters = args.NumFilters;
            self.FilterFactor = args.FilterFactor;
            self.FilterSize = args.FilterSize;
            self.Stride = args.Stride;
            self.Padding = args.Padding;
            self.Pooling = args.Pooling;
            self.NumFiltersDecoder = args.NumFiltersDecoder;
            self.FilterFactorDecoder = args.FilterFactorDecoder;
            self.FilterSizeDecoder = args.FilterSizeDecoder;
            self.StrideDecoder = args.StrideDecoder;
            self.PaddingDecoder = args.PaddingDecoder;

        end


        function [lgraph, lastLayerName] = initEncoderInputLayers( self )
            % Initialize the encoder's input layers
            arguments
                self        ConvBranchedModel       
            end
            
            if self.HasInputNormalization
                layers = featureInputLayer( self.XInputDim*self.XChannels, ...
                                            'Name', 'in', ...
                                            'Normalization', 'zscore', ...
                                            'Mean', 0, 'StandardDeviation', 1 );
            else
                layers = featureInputLayer( self.XInputDim*self.XChannels, ...
                                            'Name', 'in' );
            end
            
            if self.InputDropout > 0
                layers = [ layers; ...
                           dropoutLayer( self.InputDropout, 'Name', 'drop0' ) ];
            end

            projectionSize = [ self.XInputDim self.XChannels 1 ];
            lastLayerName = 'proj';
            layers = [ layers; ...
                       reshapeLayer( projectionSize, 'Name', lastLayerName ) ];
            
            lgraph = layerGraph( layers );
            
            end


        function [lgraph, lastLayerName] = initEncoderHiddenLayers( self, ...
                                                     lgraph, lastLayerName, offset )
            % Initialize the encoder's hidden layers
            arguments
                self            ConvBranchedModel
                lgraph   
                lastLayerName   char
                offset          double = 0
            end
            
            switch lower(self.Padding)
                case 'none'
                    pad = 0;
                case 'same'
                    pad = 'same';
            end

            numFilters = self.NumFilters;
            for i = 1:self.NumHidden

                convLayer = convolution1dLayer( self.FilterSize, ...
                                                numFilters, ...
                                                'Stride', self.Stride, ...
                                                'Padding', pad, ...
                                                'Name', ['conv' num2str(i+offset)] );

                [lgraph, lastLayerName] = self.addBlock( convLayer, ...
                                    lgraph, i + offset, lastLayerName, ...
                                    self.ReluScale, ...
                                    self.Dropout, ...
                                    self.NetNormalizationType, ...
                                    self.NetActivationType );

                numFilters = fix( numFilters*self.FilterFactor );

            end
            
            % add the output layers
            switch self.Pooling
                case 'GlobalMax'
                    poolingLayerName = ['maxPool' num2str(offset)];
                    outLayers = globalMaxPooling1dLayer( 'Name', poolingLayerName );
                case 'GlobalAvg'
                    poolingLayerName = ['avgPool' num2str(offset)];
                    outLayers = globalAveragePooling1dLayer( 'Name', poolingLayerName );
                case 'None'
                    poolingLayerName = ['fc' num2str(offset)];
                    outLayers = [];
            end
            
            % add the final layer
            if self.HasBranchedEncoder && self.HasEncoderMasking
                % set the output dimension to 1 only
                outDim = 1 + self.IsVAE;
            else
                % set the output dimension - double size if VAE        
                outDim = (1 + self.IsVAE)*self.ZDim;
            end

            outLayers = [ outLayers;
                          fullyConnectedLayer( outDim, ...
                                               'Name', ['fc' num2str(offset)] ) ];
            
            lgraph = addLayers( lgraph, outLayers );
            lgraph = connectLayers( lgraph, ...
                                    lastLayerName, poolingLayerName );
            lastLayerName = poolingLayerName;

        end


        function [lgraph, lastLayerName] = initDecoderInputLayers( self )
            % Initialize the decoder's input layers
            arguments
                self        ConvBranchedModel
            end

            lastLayerName = 'reshape';

            inLayers = [ featureInputLayer( self.ZDim, 'Name', 'in' ); ...
                         reshapeLayer( [self.ZDim 1], 'Name', lastLayerName ) ];
            
            lgraph = layerGraph( inLayers );


        end


        function [lgraph, finalLayerName] = initDecoderHiddenLayers( self, ...
                                                     lgraph, lastLayerName, offset )
            % Initialize the decoder's hidden layers
            arguments
                self            ConvBranchedModel
                lgraph   
                lastLayerName   char
                offset          double = 0
            end

            switch lower(self.PaddingDecoder)
                case 'none'
                    pad = 0;
                case 'same'
                    pad = 'same';
            end

            numFilters = self.NumFiltersDecoder*2^(self.NumHiddenDecoder-1);
            for i = 1:self.NumHiddenDecoder

                convLayer = transposedConv1dLayer( self.FilterSizeDecoder, ...
                                                   numFilters, ...
                                                   'Stride', self.StrideDecoder, ...
                                                   'Cropping', pad, ...
                                                   'Name', ['tconv' num2str(i+offset)] );

                [lgraph, lastLayerName] = self.addBlock( convLayer, ...
                                    lgraph, i + offset, lastLayerName, ...
                                    self.ReluScaleDecoder, ...
                                    self.DropoutDecoder, ...
                                    self.NetNormalizationTypeDecoder, ...
                                    self.NetActivationTypeDecoder );

                numFilters = fix( numFilters/self.FilterFactorDecoder );

            end

            combineLayerName = ['combine' num2str(i+offset)];
            finalLayerName = ['comp' num2str(offset)];
            outLayers = [ convolution1dLayer( 1, 1, 'Name', combineLayerName ); ...
                          reshapeLayer( self.XTargetDim, ...
                                        'Dims', 'CB', ...
                                        'Name', finalLayerName) ];

            lgraph = addLayers( lgraph, outLayers );
            lgraph = connectLayers( lgraph, ...
                                    lastLayerName, combineLayerName );


        end


        function self = setXTargetDim( self )
            % Calculate the decoder's output size
            arguments
                self           ConvBranchedModel
            end

            if self.HasBranchedDecoder && self.HasDecoderMasking
                outDim = self.ZDim - self.ZDimAux + 1;
            else
                outDim = self.ZDim;
            end
            
            for i = 1:self.NumHiddenDecoder
                if strcmpi(self.PaddingDecoder, 'same')
                    outDim = self.StrideDecoder*(outDim - 1) ...
                                + self.FilterSizeDecoder;
                else % valid padding
                    outDim = self.StrideDecoder*(outDim - 1)...
                                + max( self.FilterSizeDecoder, ...
                                       self.StrideDecoder );
                end
            end

            self.XTargetDim = outDim;

        end

        
    end

end


