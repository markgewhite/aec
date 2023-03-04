classdef MultiNetFCModel < FCModel
    % Subclass of a fully connected model allowing 
    % a multi-network decoder
    properties
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
            self.NumHiddenDecoder = args.NumHiddenDecoder;
            self.NumFCDecoder = args.NumFCDecoder;
            self.FCFactorDecoder = args.FCFactorDecoder;
            self.ReluScaleDecoder = args.ReluScaleDecoder;
            self.DropoutDecoder = args.DropoutDecoder;
            self.NetNormalizationTypeDecoder = args.NetNormalizationTypeDecoder;
            self.NetActivationTypeDecoder = args.NetActivationTypeDecoder;
           
        end


        function net = initDecoder( self )
            % Override the FC decoder network initialization
            arguments
                self        MultiNetFCModel
            end

            inLayer = featureInputLayer( self.ZDim, 'Name', 'in' );
            addLayer = additionLayer( self.ZDim, 'Name', 'add' );
            lgraphDec = layerGraph( inLayer );
            lgraphDec = addLayers( lgraphDec, addLayer );

            for d = 1:self.ZDim

                lastLayer = 'in';
                
                for i = 1:self.NumHiddenDecoder
    
                    nNodes = fix( self.NumFCDecoder*2^(self.FCFactorDecoder*(-self.NumHiddenDecoder+i)) );
                    if nNodes < self.ZDim
                        eid = 'FCModel:Design';
                        msg = 'Decoder hidden layer smaller than latent space.';
                        throwAsCaller( MException(eid,msg) );
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
