classdef AsymmetricFCModel < FCModel
    % Subclass of a fully connected model allowing different setting 
    % for the decoder network
    properties
        NumHiddenDecoder      % number of hidden layers for the decoder
        NumFCDecoder          % number of nodes for widest decoder layer 
        FCFactorDecoder       % log2 scaling factor subsequent layers
        ReLuScaleDecoder      % leaky ReLu scale factor
        DropoutDecoder        % hidden layer dropout rate
        NetNormalizationTypeDecoder % type of normalization
    end

    methods

        function self = AsymmetricFCModel( thisDataset, ...
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
                args.ReLuScaleDecoder   double ...
                    {mustBeInRange(args.ReLuScaleDecoder, 0, 1)} = 0.2
                args.DropoutDecoder     double ...
                    {mustBeInRange(args.DropoutDecoder, 0, 1)} = 0.0
                args.NetNormalizationTypeDecoder char ...
                    {mustBeMember( args.NetNormalizationTypeDecoder, ...
                    {'None', 'Batch', 'Layer'} )} = 'None'
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
            self.ReLuScaleDecoder = args.ReLuScaleDecoder;
            self.DropoutDecoder = args.DropoutDecoder;
            self.NetNormalizationTypeDecoder = args.NetNormalizationTypeDecoder;
           
        end


        function net = initDecoder( self )
            % Override the FC decoder network initialization
            arguments
                self        AsymmetricFCModel
            end

            layersDec = featureInputLayer( self.ZDim, 'Name', 'in' );
            
            lgraphDec = layerGraph( layersDec );
            lastLayer = 'in';
            
            for i = 1:self.NumHiddenDecoder

                nNodes = fix( self.NumFCDecoder*2^(self.FCFactorDecoder*(-self.NumHiddenDecoder+i)) );
                if nNodes < self.ZDim
                    eid = 'FCModel:Design';
                    msg = 'Decoder hidden layer smaller than latent space.';
                    throwAsCaller( MException(eid,msg) );
                end
                
                [lgraphDec, lastLayer] = FCModel.addBlock( ...
                                    lgraphDec, i, lastLayer, nNodes, ...
                                    self.ReLuScaleDecoder, ...
                                    self.DropoutDecoder, ...
                                    self.NetNormalizationTypeDecoder );

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


    end

end
