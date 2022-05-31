% ************************************************************************
% Class: lstmfcModel
%
% Subclass defining a long/short term memory encoder
% with a fully connected decoder model
%
% ************************************************************************

classdef lstmfcModel < autoencoderModel

    properties
        NumLSTMHidden           % number of LSTM hidden layers
        NumLSTMNodes            % number LSTM nodes
        LSTMFactor              % log2 scaling factor subsequent layers
        NumFCHidden             % number of FC hidden layers
        NumFCNodes              % number of nodes for widest layer
        FCFactor                % log2 scaling factor subsequent layers
        Scale                   % leaky ReLu scale factor
        InputDropout            % initial dropout rate
        Dropout                 % dropout rate
        Bidirectional           % if the network is bidirectional
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
                args.numLSTMHidden      double ...
                    {mustBeInteger, mustBePositive} = 3
                args.numLSTMNodes       double ...
                    {mustBeInteger, mustBePositive} = 16
                args.lstmFactor         double ...
                    {mustBeInteger} = 0
                args.numFCHidden        double ...
                    {mustBeInteger, mustBePositive} = 2
                args.numFCNodes         double ...
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
            self.NumLSTMHidden = args.numLSTMHidden;
            self.NumLSTMNodes = args.numLSTMNodes;
            self.LSTMFactor = args.lstmFactor;
            self.NumFCHidden = args.numFCHidden;
            self.NumFCNodes = args.numFCNodes;
            self.FCFactor = args.fcFactor;

            self.Scale = args.scale;
            self.InputDropout = args.inputDropout;
            self.Dropout = args.dropout;

            self.Bidirectional = args.bidirectional;

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
                dropoutLayer( self.InputDropout, 'Name', 'drop0' )
                ];
            
            lgraphEnc = layerGraph( layersEnc );
            lastLayer = 'drop0';
            
            for i = 1:self.NumLSTMHidden

                nNodes = fix( self.NumLSTMNodes*2^(self.LSTMFactor*(i-1)) );
                sequenceOutput = (i < self.NumLSTMHidden);

                [lgraphEnc, lastLayer] = addLSTMBlock( lgraphEnc, i, lastLayer, ...
                        nNodes, self.Bidirectional, ...
                        self.Scale, self.Dropout, sequenceOutput );

            end
            
            outLayers = fullyConnectedLayer( self.ZDim*(self.IsVAE+1), ...
                                               'Name', 'out' );
            
            lgraphEnc = addLayers( lgraphEnc, outLayers );
            lgraphEnc = connectLayers( lgraphEnc, ...
                                       lastLayer, 'out' );

            if self.IsVAE
                self.Nets.Encoder = dlnetworkVAE( lgraphEnc, ...
                                                  nDraws = self.NumVAEDraws );
            else
                self.Nets.Encoder = dlnetwork( lgraphEnc );
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
            
            for i = 1:self.NumFCHidden

                nNodes = fix( self.NumFCNodes*2^(self.FCFactor*(-self.NumFCHidden+i)) );

                [lgraphDec, lastLayer] = addFCBlock( lgraphDec, i, lastLayer, ...
                                    nNodes, self.Scale, self.Dropout );

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

            self.Nets.Decoder = dlnetwork( lgraphDec );

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

            batchSize = size( self.Nets.Encoder.State.Value{1}, 2 );
            nBatches = fix(nObs/batchSize);

            j = 1;
            for i = 1:nBatches
                dlZ(:,j:j+batchSize-1) = predict( self.Nets.Encoder, ...
                                        dlX(:,j:j+batchSize-1,:) );
                j = j+batchSize;
            end

            dlZ( :, end-batchSize+1:end ) = predict( self.Nets.Encoder, ...
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