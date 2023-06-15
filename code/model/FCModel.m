classdef FCModel < AEModel
    % Subclass defining a fully connected autoencoder model
    properties
        NumHidden             % number of hidden layers
        NumFC                 % number of nodes for widest layer
        FCFactor              % log2 scaling factor subsequent layers
        NetNormalizationType  % type of batch normalization applied
        NetActivationType     % type of nonlinear activation function
        ReluScale             % leaky ReLu scale factor
        Dropout               % hidden layer dropout rate
        InputDropout          % input dropout rate
        HasInputNormalization % apply amplitude normalization
    end

    methods

        function self = FCModel( thisDataset, ...
                                 superArgs, ...
                                 superArgs2, ...
                                 args )
            % Initialize the model
            arguments
                thisDataset     ModelDataset
                superArgs.?AEModel
                superArgs2.name     string
                superArgs2.path     string
                args.NumHidden      double ...
                    {mustBeInteger, mustBePositive} = 2
                args.NumFC          double ...
                    {mustBeInteger, mustBePositive} = 64
                args.FCFactor       double ...
                    {mustBeInteger, mustBePositive} = 2
                args.NetNormalizationType char ...
                    {mustBeMember( args.NetNormalizationType, ...
                    {'None', 'Batch', 'Layer'} )} = 'None'
                args.NetActivationType char ...
                    {mustBeMember( args.NetActivationType, ...
                    {'None', 'Tanh', 'Relu'} )} = 'Tanh'
                args.ReluScale      double ...
                    {mustBeInRange(args.ReluScale, 0, 1)} = 0.2
                args.InputDropout   double ...
                    {mustBeInRange(args.InputDropout, 0, 1)} = 0.2
                args.Dropout    double ...
                    {mustBeInRange(args.Dropout, 0, 1)} = 0.05
                args.HasInputNormalization logical = true
                args.FlattenInput   logical = true
                args.HasSeqInput    logical = false
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            superArgs2Cell = namedargs2cell( superArgs2 );

            self@AEModel( thisDataset, ...
                          superArgsCell{:}, ...
                          superArgs2Cell{:}, ...
                          FlattenInput = args.FlattenInput, ...
                          HasSeqInput = args.HasSeqInput );

            % store this class's properties
            self.NumHidden = args.NumHidden;
            self.NumFC = args.NumFC;
            self.FCFactor = args.FCFactor;
            self.NetNormalizationType = args.NetNormalizationType;
            self.NetActivationType = args.NetActivationType;
            self.ReluScale = args.ReluScale;
            self.Dropout = args.Dropout;
            self.InputDropout = args.InputDropout;
            self.HasInputNormalization = args.HasInputNormalization;
           
        end


        function net = initEncoder( self )
            % Initialize the encoder network
            arguments
                self        FCModel
            end

            if self.HasInputNormalization
                layersEnc = featureInputLayer( self.XInputDim*self.XChannels, ...
                                       'Name', 'in', ...
                                       'Normalization', 'zscore', ...
                                       'Mean', 0, 'StandardDeviation', 1 );
            else
                layersEnc = featureInputLayer( self.XInputDim*self.XChannels, ...
                                       'Name', 'in' );
            end

            if self.InputDropout > 0
                layersEnc = [ layersEnc; ...
                              dropoutLayer( self.InputDropout, 'Name', 'drop0' ) ];
                lastLayer = 'drop0';
            else
                lastLayer = 'in';
            end

            lgraphEnc = layerGraph( layersEnc );       
            
            for i = 1:self.NumHidden

                nNodes = fix( self.NumFC*2^(self.FCFactor*(1-i)) );
                if nNodes < self.ZDim
                    eid = 'FCModel:Design';
                    msg = 'Encoder hidden layer smaller than latent space.';
                    throwAsCaller( MException(eid,msg) );
                end

                [lgraphEnc, lastLayer] = self.addBlock( ...
                                    lgraphEnc, i, lastLayer, nNodes, ...
                                    self.ReluScale, ...
                                    self.Dropout, ...
                                    self.NetNormalizationType, ...
                                    self.NetActivationType );

            end
            
            outLayers = fullyConnectedLayer( 2*self.ZDim, 'Name', 'out' );
            
            lgraphEnc = addLayers( lgraphEnc, outLayers );
            lgraphEnc = connectLayers( lgraphEnc, ...
                                       lastLayer, 'out' );

            net = dlnetwork( lgraphEnc );

        end


        function net = initDecoder( self )
            % Initialize the decoder network
            arguments
                self        FCModel
            end

            layersDec = featureInputLayer( self.ZDim, 'Name', 'in' );
            
            lgraphDec = layerGraph( layersDec );
            lastLayer = 'in';
            
            for i = 1:self.NumHidden

                nNodes = fix( self.NumFC*2^(self.FCFactor*(-self.NumHidden+i)) );
                if nNodes < self.ZDim
                    eid = 'FCModel:Design';
                    msg = 'Decoder hidden layer smaller than latent space.';
                    throwAsCaller( MException(eid,msg) );
                end

                [lgraphDec, lastLayer] = self.addBlock( ...
                                    lgraphDec, i, lastLayer, nNodes, ...
                                    self.ReluScale, ...
                                    self.Dropout, ...
                                    self.NetNormalizationType, ...
                                    self.NetActivationType );

            end

            outLayers = fullyConnectedLayer( self.XTargetDim*self.XChannels, ...
                                               'Name', 'fcout' );

            if self.XChannels > 1
                outLayers = [ outLayers; 
                                reshapeLayer( [self.XTargetDim self.XChannels], ...
                                              'Name', 'reshape' ) ];
            end
            
            lgraphDec = addLayers( lgraphDec, outLayers );
            lgraphDec = connectLayers( lgraphDec, ...
                                       lastLayer, 'fcout' );

            net = dlnetwork( lgraphDec );

        end


        function dlZ = predict( encoder, X, arg )
            % Override the predict function for a fully-connected network
            % to flatten the input array
            arguments
                encoder         dlnetwork
                X               {mustBeA( X, {'dlarray', 'ModelDataset'} )}
                arg.convert     logical = true
            end

            if isa( X, 'ModelDataset' )
                dlX = self.getDLArrays( X );
            else
                dlX = X;
            end

            if size( dlX, 3 ) > 1
                CDimIdx = finddim( dlX, 'C' );
                SDimIdx = finddim( dlX, 'S' );
                BDimIdx = finddim( dlX, 'B' );
                dlX = reshape( dlX, size(dlX,SDimIdx)*size(dlX,CDimIdx), ...
                               size(dlX,BDimIdx) );
                dlX = dlarray( dlX, 'CB' );
            end

            % generate latent encodings
            dlZ = predict( encoder, dlX );

            if arg.convert
                dlZ = double(extractdata( dlZ ))';
            end

        end


        function self = setXTargetDim( self )
            % Calculate the decoder's output size
            arguments
                self           FCModel
            end

            self.XTargetDim = self.NumFC;

        end


    end

    methods (Static)

        function [ lgraph, lastLayer ] = addBlock( ...
                                        lgraph, i, lastLayer, nNodes, ...
                                        scale, dropout, normType, actType )
            % Defines block

            block = fullyConnectedLayer( nNodes, 'Name', ['fc' num2str(i)] );

            % add the specified type of normalization
            switch normType
                case 'Batch'
                    block = [ block;
                              batchNormalizationLayer( 'Name', ...
                                            ['bnorm' num2str(i)] ) ];
                case 'Layer'
                    block = [ block;
                              layerNormalizationLayer( 'Name', ...
                                            ['lnorm' num2str(i)] ) ];
            end

            % add the nonlinearity
            switch actType
                case 'Tanh'
                    block = [ block;
                      tanhLayer( 'Name', ['tanh' num2str(i)] ) ];
                case 'Relu'
                    if scale < 1
                        block = [ block;
                          leakyReluLayer( scale, 'Name', ['relu' num2str(i)] ) ];
                    end
            end

            % add dropout
            if dropout > 0
                block = [ block;
                      dropoutLayer( dropout, 'Name', ['drop' num2str(i)] ) ];
            end
        
            % connect layers at the front
            lgraph = addLayers( lgraph, block );
            lgraph = connectLayers( lgraph, ...
                                    lastLayer, ['fc' num2str(i)] );
            
            lastLayer = block(end).Name;
        
        end

        
    end

end


