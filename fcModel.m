% ************************************************************************
% Class: fcModel
%
% Subclass defining a fully connected autoencoder model
%
% ************************************************************************

classdef fcModel < autoencoderModel

    properties
        nHidden       % number of hidden layers
        nFC           % number of nodes for widest layer
        fcFactor      % log2 scaling factor subsequent layers
        scale         % leaky ReLu scale factor
        inputDropout  % input dropout rate
        dropout       % hidden layer dropout rate

    end

    methods

        function self = fcModel( XDim, XChannels, lossFcns, ...
                                 superArgs, args )
            % Initialize the model
            arguments
                XDim            double {mustBeInteger, mustBePositive}
                XChannels       double {mustBeInteger, mustBePositive}
            end
            arguments (Repeating)
                lossFcns     lossFunction
            end
            arguments
                superArgs.?autoencoderModel
                args.nHidden    double ...
                    {mustBeInteger, mustBePositive} = 2
                args.nFC        double ...
                    {mustBeInteger, mustBePositive} = 64
                args.fcFactor   double ...
                    {mustBeInteger, mustBePositive} = 2
                args.scale      double ...
                    {mustBeInRange(args.scale, 0, 1)} = 0.2
                args.inputDropout   double ...
                    {mustBeInRange(args.inputDropout, 0, 1)} = 0.2
                args.dropout    double ...
                    {mustBeInRange(args.dropout, 0, 1)} = 0.05
                args.isVAE         logical = false
                args.nVAEDraws     double ...
                    {mustBeInteger, mustBePositive} = 1
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@autoencoderModel( XDim, ...
                                          XChannels, ...
                                          lossFcns{:}, ...
                                          superArgsCell{:}, ...
                                          hasSeqInput = false );

            % store this class's properties
            self.nHidden = args.nHidden;
            self.nFC = args.nFC;
            self.fcFactor = args.fcFactor;
            self.scale = args.scale;
            self.inputDropout = args.inputDropout;
            self.dropout = args.dropout;
            self.isVAE = args.isVAE;

            % initialize the networks
            self = initEncoder( self, args.nVAEDraws );
            self = initDecoder( self );
            
        end


        function self = initEncoder( self, nVAEDraws )
            % Initialize the encoder network

            layersEnc = [ featureInputLayer( self.XDim, 'Name', 'in', ...
                                   'Normalization', 'zscore', ...
                                   'Mean', 0, 'StandardDeviation', 1 ) 
                          dropoutLayer( self.inputDropout, 'Name', 'drop0' ) ];
            
            for i = 1:self.nHidden
                nNodes = fix( self.nFC*2^(self.fcFactor*(1-i)) );
                layersEnc = [ layersEnc; ...
                    fullyConnectedLayer( nNodes, 'Name', ['fc' num2str(i)] )
                    batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                    leakyReluLayer( self.scale, ...
                                    'Name', ['relu' num2str(i)] )
                    dropoutLayer( self.dropout, 'Name', ...
                                                     ['drop' num2str(i)] )
                    ]; %#ok<*AGROW> 
            end
            
            layersEnc = [ layersEnc; ...
                  fullyConnectedLayer( self.ZDim*(self.isVAE+1), ...
                                       'Name', 'out' ) ];
            
            lgraphEnc = layerGraph( layersEnc );

            if self.isVAE
                self.nets.encoder = dlnetworkVAE( lgraphEnc, ...
                                                  nDraws = nVAEDraws );
            else
                self.nets.encoder = dlnetwork( lgraphEnc );
            end

        end


        function self = initDecoder( self )
            % Initialize the decoder network

            layersDec = featureInputLayer( self.ZDim, 'Name', 'in' );
            
            for i = 1:self.nHidden
                nNodes = fix( self.nFC*2^(self.fcFactor*(-self.nHidden+i)) );
                layersDec = [ layersDec; ...
                    fullyConnectedLayer( nNodes, 'Name', ['fc' num2str(i)] )
                    batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                    leakyReluLayer( self.scale, ...
                                    'Name', ['relu' num2str(i)] )           
                    ]; %#ok<*AGROW> 
            end

            layersDec = [ layersDec; ...
                          fullyConnectedLayer( self.XOutDim*self.XChannels, ...
                                               'Name', 'fcout' ) ];

            if self.XChannels > 1
                layersDec = [ layersDec; 
                                reshapeLayer( [self.XOutDim self.XChannels], ...
                                              'Name', 'reshape' ) ];
            end
            
            lgraphDec = layerGraph( layersDec );

            self.nets.decoder = dlnetwork( lgraphDec );

        end


    end

end