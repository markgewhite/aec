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

        function self = fcModel( XDim, XOutputDim, XChannels, ZDim, CDim, ...
                                   lossFcns, superArgs, args )
            % Initialize the model
            arguments
                XDim            double {mustBeInteger, mustBePositive}
                XOutputDim      double {mustBeInteger, mustBePositive}
                XChannels       double {mustBeInteger, mustBePositive}
                ZDim            double {mustBeInteger, mustBePositive}
                CDim            double {mustBeInteger, mustBePositive}
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
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@autoencoderModel( XDim, ...
                                          XOutputDim, ...
                                          XChannels, ...
                                          ZDim, ...
                                          CDim, ...
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

            % initialize the networks
            self = initEncoder( self );
            self = initDecoder( self );
            
        end


        function self = initEncoder( self )
            % Initialize the encoder network
            arguments
                self        fcModel
            end

            layersEnc = [ featureInputLayer( self.XDim*self.XChannels, ...
                                   'Name', 'in', ...
                                   'Normalization', 'zscore', ...
                                   'Mean', 0, 'StandardDeviation', 1 ) 
                          dropoutLayer( self.inputDropout, 'Name', 'drop0' ) ];

            lgraphEnc = layerGraph( layersEnc );       
            lastLayer = 'drop0';
            
            for i = 1:self.nHidden

                nNodes = fix( self.nFC*2^(self.fcFactor*(1-i)) );

                [lgraphEnc, lastLayer] = addBlock( lgraphEnc, i, lastLayer, ...
                                    nNodes, self.scale, self.dropout );

            end
            
            outLayers = fullyConnectedLayer( self.ZDim*(self.isVAE+1), ...
                                               'Name', 'out' );
            
            lgraphEnc = addLayers( lgraphEnc, outLayers );
            lgraphEnc = connectLayers( lgraphEnc, ...
                                       lastLayer, 'out' );

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
                self        fcModel
            end

            layersDec = featureInputLayer( self.ZDim, 'Name', 'in' );
            
            lgraphDec = layerGraph( layersDec );
            lastLayer = 'in';
            
            for i = 1:self.nHidden

                nNodes = fix( self.nFC*2^(self.fcFactor*(-self.nHidden+i)) );

                [lgraphDec, lastLayer] = addBlock( lgraphDec, i, lastLayer, ...
                                    nNodes, self.scale, self.dropout );

            end

            outLayers = fullyConnectedLayer( self.XOutputDim*self.XChannels, ...
                                               'Name', 'fcout' );

            if self.XChannels > 1
                outLayers = [ outLayers; 
                                reshapeLayer( [self.XOutputDim self.XChannels], ...
                                              'Name', 'reshape' ) ];
            end
            
            lgraphDec = addLayers( lgraphDec, outLayers );
            lgraphDec = connectLayers( lgraphDec, ...
                                       lastLayer, 'fcout' );

            self.nets.decoder = dlnetwork( lgraphDec );

        end


        function [ dlXHat, dlZ, state ] = forward( self, encoder, decoder, dlX )
            % Forward-run the fully-connected network
            % by first flattening the input array
            arguments
                self        fcModel
                encoder     dlnetwork
                decoder     dlnetwork
                dlX         dlarray
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
            [ dlZ, state.encoder ] = forward( encoder, dlX );
    
            % reconstruct curves from latent codes
            [ dlXHat, state.decoder ] = forward( decoder, dlZ );

        end

    end


    methods (Static )

        function dlZ = encode( self, X, arg )
            % Encode features Z from X using the model
            % by first flattening the input array
            arguments
                self            autoencoderModel
                X
                arg.convert     logical = true
            end

            if isa( X, 'modelDataset' )
                dlX = X.getDLInput( self.XDimLabels );
            elseif isa( X, 'dlarray' )
                dlX = X;
            else
                eid = 'Autoencoder:NotValidX';
                msg = 'The input data should be a modelDataset or a dlarray.';
                throwAsCaller( MException(eid,msg) );
            end

            if size( dlX, 3 ) > 1
                % flatten the input array
                dlX = reshape( dlX, size(dlX,1)*size(dlX,2), size(dlX,3) );
                dlX = dlarray( dlX, 'CB' );
            end

            dlZ = predict( self.nets.encoder, dlX );

            if arg.convert
                dlZ = double(extractdata( dlZ ))';
            end

        end

    end

end


function [ lgraph, lastLayer ] = addBlock( lgraph, i, lastLayer, ...
                                           nNodes, scale, dropout )

    % define block
    block = [   fullyConnectedLayer( nNodes, ...
                                'Name', ['fc' num2str(i)] )
                layerNormalizationLayer( 'Name', ...
                                ['lnorm' num2str(i)] )
                leakyReluLayer( scale, ...
                                'Name', ['relu' num2str(i)] )
                spatialDropoutLayer( dropout, ...
                                     'Name', ['drop' num2str(i)] )
                ];

    % connect layers at the front
    lgraph = addLayers( lgraph, block );
    lgraph = connectLayers( lgraph, ...
                            lastLayer, ['fc' num2str(i)] );
    
    lastLayer = ['drop' num2str(i)];

end