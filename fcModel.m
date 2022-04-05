% ************************************************************************
% Class: fcModel
%
% Subclass defining a fully connected autoencoder model
%
% ************************************************************************

classdef fcModel < aeModel

    properties
        nHidden       % number of hidden layers
        nFC           % number of nodes for widest layer
        fcFactor      % log2 scaling factor subsequent layers
        scale         % leaky ReLu scale factor
        dropout       % dropout rate

    end

    methods

        function self = fcModel( superArgs, args )
            % Initialize the model
            arguments
                superArgs.?aeModel
                args.nHidden    double ...
                    {mustBeInteger, mustBePositive} = 2
                args.nFC        double ...
                    {mustBeInteger, mustBePositive} = 64
                args.fcFactor   double ...
                    {mustBeInteger, mustBePositive} = 2
                args.scale      double ...
                    {mustBeInRange(args.scale, 0, 1)} = 0.2
                args.dropout    double ...
                    {mustBeInRange(args.dropout, 0, 1)} = 0.1
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@aeModel( superArgsCell{:} );

            % store this class's properties
            self.nHidden = args.nHidden;
            self.nFC = args.nFC;
            self.fcFactor = args.fcFactor;
            self.scale = args.scale;
            self.dropout0 = args.dropout;

            % define the encoder network
            layersEnc = [ featureInputLayer( self.XDim, 'Name', 'in', ...
                                   'Normalization', 'zscore', ...
                                   'Mean', 0, 'StandardDeviation', 1 ) 
                          dropoutLayer( self.dropout, 'Name', 'drop0' ) ];
            
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
                  fullyConnectedLayer( self.ZDim, 'Name', 'out' ) ];
            
            lgraphEnc = layerGraph( layersEnc );
            self.nets.encoder = dlnetwork( lgraphEnc );


            % define the decoder network          
            layersDec = featureInputLayer( self.ZDim, 'Name', 'in' );
            
            for i = 1:paramDec.nHidden
                nNodes = fix( self.nFC*2^(self.fcFactor*(-self.nHidden+i)) );
                layersDec = [ layersDec; ...
                    fullyConnectedLayer( nNodes, 'Name', ['fc' num2str(i)] )
                    batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                    leakyReluLayer( self.scale, ...
                                    'Name', ['relu' num2str(i)] )           
                    ]; %#ok<*AGROW> 
            end

            layersDec = [ layersDec; ...
                          fullyConnectedLayer( self.XDim*self.XChannels, 'Name', 'fcout' )
                          reshapeLayer( [self.XDim self.XChannels], 'Name', 'out' ) ];
            
            lgraphDec = layerGraph( layersDec );
            self.nets.decoder = dlnetwork( lgraphDec );
            
            
        end

    end

end