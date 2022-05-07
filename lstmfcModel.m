% ************************************************************************
% Class: lstmfcModel
%
% Subclass defining a long/short term memory encoder
% with a fully connected decoder model
%
% ************************************************************************

classdef lstmfcModel < autoencoderModel

    properties
        nHidden                 % number of hidden layers
        nFC                     % number of nodes for widest layer
        fcFactor                % log2 scaling factor subsequent layers
        nLSTMUnits              % number LSTM nodes
        scale                   % leaky ReLu scale factor
        inputDropout            % initial dropout rate
        dropout                 % dropout rate
        bidirectional           % if the network is bidirectional
    end

    methods

        function self = lstmfcModel( XDim, XOutputDim, XChannels, ...
                                   lossFcns, superArgs, args )
            % Initialize the model
            arguments
                XDim            double {mustBeInteger, mustBePositive}
                XOutputDim      double {mustBeInteger, mustBePositive}
                XChannels       double {mustBeInteger, mustBePositive}
            end
            arguments (Repeating)
                lossFcns     lossFunction
            end
            arguments
                superArgs.?autoencoderModel
                args.nHidden    double ...
                    {mustBeInteger, mustBePositive} = 1
                args.nFC        double ...
                    {mustBeInteger, mustBePositive} = 64
                args.fcFactor   double ...
                    {mustBeInteger, mustBePositive} = 2
                args.nLSTMUnits       double ...
                    {mustBeInteger, mustBePositive} = 50
                args.scale              double ...
                    {mustBeInRange(args.scale, 0, 1)} = 0.2
                args.inputDropout       double ...
                    {mustBeInRange(args.inputDropout, 0, 1)} = 0.10
                args.dropout            double ...
                    {mustBeInRange(args.dropout, 0, 1)} = 0.05
                args.reverseDecoding    logical = true
                args.bidirectional      logical = false
                args.scheduleSampling   logical = true
                args.samplingRateIncrement double ...
                    {mustBePositive} = 0.005
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@autoencoderModel( XDim, ...
                                          XOutputDim, ...
                                          XChannels, ...
                                          lossFcns{:}, ...
                                          superArgsCell{:}, ...
                                          hasSeqInput = true, ...
                                          isVAE = false );


            % store this class's properties
            self.nHidden = args.nHidden;
            self.nFC = args.nFC;
            self.fcFactor = args.fcFactor;

            self.nLSTMUnits = args.nLSTMUnits;

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
            
            for i = 1:self.nHidden

                [lgraphEnc, lastLayer] = addLSTMBlock( lgraphEnc, i, lastLayer, ...
                        self.nLSTMUnits, self.bidirectional, ...
                        self.scale, self.dropout );

            end
            
            outLayers = fullyConnectedLayer( self.ZDim*(self.isVAE+1), ...
                                               'Name', 'out' );
            
            lgraphEnc = addLayers( lgraphEnc, outLayers );
            lgraphEnc = connectLayers( lgraphEnc, ...
                                       lastLayer, 'out' );
                   
            self.nets.encoder = dlnetwork( lgraphEnc );

        end


        function self = initDecoder( self )
            % Initialize the decoder network
            arguments
                self        lstmfcModel
            end

            layersDec = featureInputLayer( self.ZDim, 'Name', 'in' );
            
            lgraphDec = layerGraph( layersDec );
            lastLayer = 'in';
            
            for i = 1:self.nHidden

                nNodes = fix( self.nFC*2^(self.fcFactor*(-self.nHidden+i)) );

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


    end


    methods (Static)

        function dlZ = encode( self, X, arg )
            % Encode features Z from X using the model
            % overriding the autoencoder encode method
            arguments
                self            autoencoderModel
                X
                arg.convert     logical = true
            end

            if isa( X, 'modelDataset' )
                dlX = X.getInput;
            elseif isa( X, 'dlarray' )
                dlX = X;
            else
                eid = 'Autoencoder:NotValidX';
                msg = 'The input data should be a modelDataset or a dlarray.';
                throwAsCaller( MException(eid,msg) );
            end

            batchSize = size( self.nets.encoder.State.Value{1}, 2 );
            nObs = size( dlX, 2 );
            nBatches = fix(nObs/batchSize);
            dlZ = dlarray( zeros( self.ZDim, nBatches*batchSize ), 'CB' );

            j = 1;
            for i = 1:nBatches
                dlZ(:,j:j+batchSize-1) = predict( self.nets.encoder, ...
                                        dlX(:,j:j+batchSize-1,:) );
                j = j+batchSize;
            end

            if arg.convert
                dlZ = double(extractdata( dlZ ))';
            end

        end

    end


end


function [ lgraph, lastLayer ] = addLSTMBlock( lgraph, i, lastLayer, ...
                               nNodes, bidirectional, scale, dropout )

    % define block
    if bidirectional
        block = bilstmLayer( nNodes, ...
                            'OutputMode', 'last', ...
                             'Name', ['lstm' num2str(i)] );
    else
        block = lstmLayer( nNodes, ...
                            'OutputMode', 'last', ...
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