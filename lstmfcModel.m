% ************************************************************************
% Class: lstmfcModel
%
% Subclass defining a long/short term memory encoder
% with a fully connected decoder model
%
% ************************************************************************

classdef lstmfcModel < autoencoderModel

    properties
        nLSTMHidden             % number of LSTM hidden layers
        nLSTMNodes              % number LSTM nodes
        lstmFactor              % log2 scaling factor subsequent layers
        nFCHidden               % number of FC hidden layers
        nFCNodes                % number of nodes for widest layer
        fcFactor                % log2 scaling factor subsequent layers
        scale                   % leaky ReLu scale factor
        inputDropout            % initial dropout rate
        dropout                 % dropout rate
        bidirectional           % if the network is bidirectional
    end

    methods

        function self = lstmfcModel( XDim, XOutputDim, XChannels, ZDim, CDim, ...
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
                args.nLSTMHidden        double ...
                    {mustBeInteger, mustBePositive} = 3
                args.nLSTMNodes         double ...
                    {mustBeInteger, mustBePositive} = 16
                args.lstmFactor         double ...
                    {mustBeInteger} = 0
                args.nFCHidden          double ...
                    {mustBeInteger, mustBePositive} = 2
                args.nFCNodes           double ...
                    {mustBeInteger, mustBePositive} = 64
                args.fcFactor           double ...
                    {mustBeInteger} = 2
                args.scale              double ...
                    {mustBeInRange(args.scale, 0, 1)} = 0.2
                args.inputDropout       double ...
                    {mustBeInRange(args.inputDropout, 0, 1)} = 0.10
                args.dropout            double ...
                    {mustBeInRange(args.dropout, 0, 1)} = 0.05
                args.bidirectional      logical = false
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
                                          hasSeqInput = true, ...
                                          isVAE = false );


            % store this class's properties
            self.nLSTMHidden = args.nLSTMHidden;
            self.nLSTMNodes = args.nLSTMNodes;
            self.lstmFactor = args.lstmFactor;
            self.nFCHidden = args.nFCHidden;
            self.nFCNodes = args.nFCNodes;
            self.fcFactor = args.fcFactor;

            self.scale = args.scale;
            self.inputDropout = args.inputDropout;
            self.dropout = args.dropout;

            self.bidirectional = args.bidirectional;

            % initialize the networks
            self = initEncoder( self );
            self = initDecoder( self );

        end


        function self = initEncoder( self )
            % Initialize the encoder network
            arguments
                self        lstmfcModel
            end

            layersEnc = [ ...
                sequenceInputLayer( self.XChannels, 'Name', 'in', ...
                                   'Normalization', 'zscore', ...
                                   'Mean', 0, 'StandardDeviation', 1 )
                dropoutLayer( self.inputDropout, 'Name', 'drop0' )
                ];
            
            lgraphEnc = layerGraph( layersEnc );
            lastLayer = 'drop0';
            
            for i = 1:self.nLSTMHidden

                nNodes = fix( self.nLSTMNodes*2^(self.lstmFactor*(i-1)) );
                sequenceOutput = (i < self.nLSTMHidden);

                [lgraphEnc, lastLayer] = addLSTMBlock( lgraphEnc, i, lastLayer, ...
                        nNodes, self.bidirectional, ...
                        self.scale, self.dropout, sequenceOutput );

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
                self        lstmfcModel
            end

            layersDec = featureInputLayer( self.ZDim, 'Name', 'in' );
            
            lgraphDec = layerGraph( layersDec );
            lastLayer = 'in';
            
            for i = 1:self.nFCHidden

                nNodes = fix( self.nFCNodes*2^(self.fcFactor*(-self.nFCHidden+i)) );

                [lgraphDec, lastLayer] = addFCBlock( lgraphDec, i, lastLayer, ...
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


        function dlZ = encode( self, X, arg )
            % Encode features Z from X using the model
            % overriding the autoencoder encode method
            % because lstm models must be given the same number 
            % of observations they were trained with, the batch size
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

            nObs = size( dlX, 2 );
            dlZ = dlarray( zeros( self.ZDim, nObs ), 'CB' );

            batchSize = size( self.nets.encoder.State.Value{1}, 2 );
            nBatches = fix(nObs/batchSize);

            j = 1;
            for i = 1:nBatches
                dlZ(:,j:j+batchSize-1) = predict( self.nets.encoder, ...
                                        dlX(:,j:j+batchSize-1,:) );
                j = j+batchSize;
            end

            dlZ( :, end-batchSize+1:end ) = predict( self.nets.encoder, ...
                                    dlX( :, end-batchSize+1:end, : ) );

            if arg.convert
                dlZ = double(extractdata( dlZ ))';
            end

        end

    end


end


function [ lgraph, lastLayer ] = addLSTMBlock( lgraph, i, lastLayer, ...
                       nNodes, bidirectional, scale, dropout, seqOutput )

    % define block
    if seqOutput
        outputMode = 'sequence';
    else
        outputMode = 'last';
    end

    if bidirectional
        block = bilstmLayer( nNodes, ...
                            'OutputMode', outputMode, ...
                             'Name', ['lstm' num2str(i)] );
    else
        block = lstmLayer( nNodes, ...
                            'OutputMode', outputMode, ...
                             'Name', ['lstm' num2str(i)] );
    end

    block = [   block;
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
                            lastLayer, ['lstm' num2str(i)] );
    
    lastLayer = ['drop' num2str(i)];

end


function [ lgraph, lastLayer ] = addFCBlock( lgraph, i, lastLayer, ...
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