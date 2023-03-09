classdef MultiNetFCModel < FCModel
    % Subclass of a fully connected model allowing 
    % a multi-network decoder
    properties
        HasBranchedEncoder    % whether to have branching in the encoder
        NumHiddenDecoder      % number of hidden layers
        NumFCDecoder          % number of nodes for widest layer
        FCFactorDecoder       % log2 scaling factor subsequent layers
        NetNormalizationTypeDecoder  % type of batch normalization applied
        NetActivationTypeDecoder     % type of nonlinear activation function
        ReluScaleDecoder             % leaky ReLu scale factor
        DropoutDecoder        % hidden layer dropout rate
    end

    methods

        function self = MultiNetFCModel( thisDataset, ...
                                           superArgs, ...
                                           superArgs2, ...
                                           args )
            % Initialize the model
            arguments
                thisDataset             ModelDataset
                superArgs.?FCModel
                superArgs2.name         string
                superArgs2.path         string
                args.HasBranchedEncoder logical = false
                args.NumHiddenDecoder   double ...
                    {mustBeInteger, mustBePositive} = 1
                args.NumFCDecoder       double ...
                    {mustBeInteger, mustBePositive} = 128
                args.FCFactorDecoder    double ...
                    {mustBeInteger, mustBePositive} = 1
                args.ReluScaleDecoder   double ...
                    {mustBeInRange(args.ReluScaleDecoder, 0, 1)} = 0.2
                args.DropoutDecoder     double ...
                    {mustBeInRange(args.DropoutDecoder, 0, 1)} = 0.0
                args.NetNormalizationTypeDecoder char ...
                    {mustBeMember( args.NetNormalizationTypeDecoder, ...
                    {'None', 'Batch', 'Layer'} )} = 'None'
                args.NetActivationTypeDecoder char ...
                    {mustBeMember( args.NetActivationTypeDecoder, ...
                    {'None', 'Tanh', 'Relu'} )} = 'Tanh'
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            superArgs2Cell = namedargs2cell( superArgs2 );

            self@FCModel( thisDataset, ...
                          superArgsCell{:}, ...
                          superArgs2Cell{:} );

            % store this class's properties
            self.HasBranchedEncoder = args.HasBranchedEncoder;
            self.NumHiddenDecoder = args.NumHiddenDecoder;
            self.NumFCDecoder = args.NumFCDecoder;
            self.FCFactorDecoder = args.FCFactorDecoder;
            self.ReluScaleDecoder = args.ReluScaleDecoder;
            self.DropoutDecoder = args.DropoutDecoder;
            self.NetNormalizationTypeDecoder = args.NetNormalizationTypeDecoder;
            self.NetActivationTypeDecoder = args.NetActivationTypeDecoder;
           
        end


        function net = initEncoder( self )
            % Override the FC encoder network initialization
            arguments
                self        MultiNetFCModel
            end

            if self.HasBranchedEncoder

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
                    lastLayer0 = 'drop0';
                else
                    lastLayer0 = 'in';
                end
    
                lgraphEnc = layerGraph( layersEnc );
                lgraphEnc = addLayers( lgraphEnc, ...
                                       additionLayer( self.ZDimAux, 'Name', 'add' ) );

                mask = [false( self.ZDimAux, 1 );
                        true( self.ZDim - self.ZDimAux, 1 )];
                
                for d = 1:self.ZDimAux
    
                    lastLayer = lastLayer0;
    
                    for i = 1:self.NumHidden
        
                        nNodes = fix( self.NumFC*2^(self.FCFactor*(1-i)) );
                        if nNodes < self.ZDim
                            eid = 'FCModel:Design';
                            msg = 'Encoder hidden layer smaller than latent space.';
                            throwAsCaller( MException(eid,msg) );
                        end
        
                        [lgraphEnc, lastLayer] = self.addBlock( ...
                                            lgraphEnc, 100*d+i, lastLayer, nNodes, ...
                                            self.ReluScale, ...
                                            self.Dropout, ...
                                            self.NetNormalizationType, ...
                                            self.NetActivationType );
        
                    end
                    
                    maskD = mask;
                    maskD(d) = true;
                    outLayers = [fullyConnectedLayer( self.ZDim, 'Name', ...
                                                    ['fcout' num2str(100*d)] );
                                 maskLayer( maskD, ...
                                            'ReduceDim', false, ...
                                            'Name', ['mask' num2str(100*d)] )];
                    
                    lgraphEnc = addLayers( lgraphEnc, outLayers );
                    lgraphEnc = connectLayers( lgraphEnc, ...
                                               lastLayer, ['fcout' num2str(100*d)] );
    
                    lgraphEnc = connectLayers( lgraphEnc, ...
                               ['mask' num2str(100*d)], ...
                               ['add/in' num2str(d)] );
    
                end
    
                net = dlnetwork( lgraphEnc );

            else
                net = initEncoder@FCModel( self );

            end

        end


        function net = initDecoder( self )
            % Override the FC decoder network initialization
            arguments
                self        MultiNetFCModel
            end

            mask = [false( self.ZDimAux, 1 );
                    true( self.ZDim - self.ZDimAux, 1 )];
            inLayer = featureInputLayer( self.ZDim, 'Name', 'in' );
            lgraphDec = layerGraph( inLayer );
            lgraphDec = addLayers( lgraphDec, ...
                                   additionLayer( self.ZDimAux, 'Name', 'add' ) );

            for d = 1:self.ZDimAux

                maskD = mask;
                maskD(d) = true;
                lgraphDec = addLayers( lgraphDec, ...
                                maskLayer( maskD, ...
                                           'ReduceDim', true, ...
                                           'Name', ['mask' num2str(100*d)] ));
                lgraphDec = connectLayers( lgraphDec, ...
                                           'in', ['mask' num2str(100*d)] );

                lastLayer = ['mask' num2str(100*d)];
                
                for i = 1:self.NumHiddenDecoder
                    
                    if i==1
                        nNodes = sum( maskD );
                    else
                        nNodes = fix( self.NumFCDecoder*2^(self.FCFactorDecoder*(-self.NumHiddenDecoder+i)) );
                        if nNodes < self.ZDim
                            eid = 'FCModel:Design';
                            msg = 'Decoder hidden layer smaller than latent space.';
                            throwAsCaller( MException(eid,msg) );
                        end
                    end
                    
                    [lgraphDec, lastLayer] = FCModel.addBlock( ...
                                        lgraphDec, 100*d+i, lastLayer, nNodes, ...
                                        self.ReluScaleDecoder, ...
                                        self.DropoutDecoder, ...
                                        self.NetNormalizationTypeDecoder, ...
                                        self.NetActivationType );
    
                end
    
                outLayers = fullyConnectedLayer( self.XTargetDim*self.XChannels, ...
                                                   'Name', ['fcout' num2str(100*d)] );
    
                if self.XChannels > 1
                    outLayers = [ outLayers; 
                                    reshapeLayer( [self.XTargetDim self.XChannels], ...
                                                  'Name', 'reshape' ) ];
                end
            
                lgraphDec = addLayers( lgraphDec, outLayers );
            
                lgraphDec = connectLayers( lgraphDec, ...
                                           lastLayer, ['fcout' num2str(100*d)] );
                lgraphDec = connectLayers( lgraphDec, ...
                                           ['fcout' num2str(100*d)], ...
                                           ['add/in' num2str(d)] );
            
            end

            net = dlnetwork( lgraphDec );

        end


    end

end
