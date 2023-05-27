classdef BranchedModel < VAEModel
    % A branched variational autoencoder
    % Subclasses define the hidden layers of the encoder and decoder networks
    properties
        HasInputNormalization % whether to apply amplitude normalization
        NumHidden             % number of hidden layers in the encoder
        NumHiddenDecoder      % number of hidden layers in the decoder
        NetNormalizationType  % type of batch normalization applied
        NetActivationType     % type of nonlinear activation function
        ReluScale             % leaky ReLu scale factor
        Dropout               % hidden layer dropout rate
        InputDropout          % input dropout rate
        HasBranchedEncoder    % whether to have branching in the encoder
        HasEncoderMasking     % whether to mask Z outputs from the encoder
        HasBranchedDecoder    % whether to have branching in the decoder
        HasDecoderMasking     % whether to mask Z inputs to the decoder
    end

    methods

        function self = BranchedModel( thisDataset, ...
                                           superArgs, ...
                                           superArgs2, ...
                                           args )
            % Initialize the model
            arguments
                thisDataset                 ModelDataset
                superArgs.?VAEModel
                superArgs2.name             string
                superArgs2.path             string
                args.HasInputNormalization  logical = true
                args.NumHidden              double ...
                    {mustBeInteger, mustBePositive} = 2
                args.NumHiddenDecoder       double ...
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
                args.HasBranchedEncoder     logical = false
                args.HasBranchedDecoder     logical = true
                args.HasEncoderMasking      logical = false
                args.HasDecoderMasking      logical = true
                args.FlattenInput           logical = true
                args.HasSeqInput            logical = false
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            superArgs2Cell = namedargs2cell( superArgs2 );

            self@VAEModel( thisDataset, ...
                           superArgsCell{:}, ...
                           superArgs2Cell{:}, ...
                           FlattenInput = args.FlattenInput, ...
                           HasSeqInput = args.HasSeqInput );

            % store this class's properties
            self.HasInputNormalization = args.HasInputNormalization;
            self.NumHidden = args.NumHidden;
            self.NumHiddenDecoder = args.NumHiddenDecoder;
            self.NetNormalizationType = args.NetNormalizationType;
            self.NetActivationType = args.NetActivationType;
            self.ReluScale = args.ReluScale;
            self.Dropout = args.Dropout;
            self.InputDropout = args.InputDropout;
            self.HasBranchedEncoder = args.HasBranchedEncoder;
            self.HasEncoderMasking = args.HasEncoderMasking;
            self.HasBranchedDecoder = args.HasBranchedDecoder;
            self.HasDecoderMasking = args.HasDecoderMasking;
           
        end


        function net = initEncoder( self )
            % Override the FC encoder network initialization
            arguments
                self        BranchedModel
            end

            [lgraph, lastInputLayer] = self.initEncoderInputLayers;

            if self.HasBranchedEncoder
                lgraph = addLayers( lgraph, ...
                                    additionLayer( self.ZDimAux, 'Name', 'add' ) );
                if self.HasEncoderMasking
                    mask = [false( self.ZDimAux, 1 );
                            true( self.ZDim - self.ZDimAux, 1 )];
                end
                dRange = 1:self.ZDimAux;

            else
                dRange = 0;

            end

            for d = dRange

                [lgraph, lastLayerName] = self.initEncoderHiddenLayers( lgraph, lastInputLayer, d*100 );
               
                if self.HasBranchedEncoder && self.HasEncoderMasking
                    maskD = mask;
                    maskD(d) = true;
                    finalLayerName = ['mask' num2str(100*d)];
                    lgraph = addLayers( lgraph, ...
                                        maskLayer( maskD, ...
                                                   'ReduceDim', false, ...
                                                    'Name', finalLayerName ) );
                else
                    finalLayerName = lastLayerName;

                end

                if self.HasBranchedEncoder
                    lgraph = connectLayers( lgraph, ...
                                            finalLayerName, ...
                                            ['add/in' num2str(d)] );
                end

            end

            net = dlnetwork( lgraph );

        end


        function net = initDecoder( self )
            % Override the FC decoder network initialization
            arguments
                self        BranchedModel
            end

            [lgraph, inputLayerName] = initDecoderInputLayers( self );
            
            if self.HasBranchedDecoder && self.ZDimAux>1

                mask = [false( self.ZDimAux, 1 );
                        true( self.ZDim - self.ZDimAux, 1 )];
                lgraph = addLayers( lgraph, ...
                                    additionLayer( self.ZDimAux, 'Name', 'add' ) );

                dRange = 1:self.ZDimAux;

            else
                dRange = 0;

            end

            for d = dRange

                if self.HasDecoderMasking
                    maskD = mask;
                    maskD(d) = true;
                    lgraph = addLayers( lgraph, ...
                                        maskLayer( maskD, ...
                                                   'ReduceDim', true, ...
                                                   'Name', ['mask' num2str(100*d)] ));
                    lastLayerName = ['mask' num2str(100*d)];
                    lgraph = connectLayers( lgraph, ...
                                            inputLayerName, lastLayerName );
                
                end

                [lgraph, lastLayerName] = self.initDecoderHiddenLayers( lgraph, lastLayerName, d*100 );
    
                if self.XChannels > 1
                    finalLayerName = ['reshape' num2str(100*d)];
                    lgraph = [ lgraph;
                               reshapeLayer( [ self.XTargetDim self.XChannels], ...
                                              'Name', finalLayerName ) ]; %#ok<AGROW> 
                else
                    finalLayerName = lastLayerName;
                end

                if self.HasBranchedDecoder
                    lgraph = connectLayers( lgraph, ...
                                            finalLayerName, ...
                                            ['add/in' num2str(d)] );
                end
            
            end

            net = dlnetwork( lgraph );

        end
        

    end


    methods (Abstract)

        % abstract methods to be defined by subclasses

        initEncoderInputLayers;

        initEncoderHiddenLayers;
        
        initDecoderInputLayers;

        initDecoderHiddenLayers;

       
    end


    methods (Static)

        function [ lgraph, lastLayer ] = addBlock( firstLayer, ...
                                        lgraph, i, precedingLayer,  ...
                                        scale, dropout, normType, actType )
            
            % initialize the block
            block = firstLayer;

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
            lgraph = connectLayers( lgraph, precedingLayer, firstLayer.Name );
            
            lastLayer = block(end).Name;
        
        end

    end
    

end
