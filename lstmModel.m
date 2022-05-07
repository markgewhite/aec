% ************************************************************************
% Class: lstmModel
%
% Subclass defining a long/short term memory autoencoder model
%
% ************************************************************************

classdef lstmModel < autoencoderModel

    properties
        nHiddenUnits            % number LSTM nodes
        scale                   % leaky ReLu scale factor
        inputDropout            % initial dropout rate
        dropout                 % dropout rate
        reverseDecoding         % whether to reverse the order when decoding
        bidirectional           % if the network is bidirectional
        scheduleSampling        % whether to perform schedule sampling
        samplingRateIncrement   % sampling rate increment per epoch
    end

    methods

        function self = lstmModel( XDim, XOutputDim, XChannels, ...
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
                args.nHiddenUnits       double ...
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
            self.nHiddenUnits = args.nHiddenUnits;
            self.scale = args.scale;
            self.inputDropout = args.inputDropout;
            self.dropout = args.dropout;

            self.reverseDecoding = args.reverseDecoding;
            self.bidirectional = args.bidirectional;
            self.scheduleSampling = args.scheduleSampling;
            self.samplingRateIncrement = args.samplingRateIncrement;

            % initialize the networks
            self = initEncoder( self );
            self = initDecoder( self );

        end


        function self = initEncoder( self )
            % Initialize the encoder network
            arguments
                self        lstmModel
            end

            layersEnc = [ ...
                sequenceInputLayer( self.XChannels, 'Name', 'in', ...
                                   'Normalization', 'zscore', ...
                                   'Mean', 0, 'StandardDeviation', 1 )
                dropoutLayer( self.inputDropout, 'Name', 'drop0' )
                ];
            
            if self.bidirectional
                layersEnc = [ layersEnc; ...
                    bilstmLayer( self.nHiddenUnits, ...
                                    'OutputMode', 'last', ...
                                     'Name', 'lstm' ) ];
            else
                layersEnc = [ layersEnc; ...
                    lstmLayer( self.nHiddenUnits, ...
                                    'OutputMode', 'last', ...
                                     'Name', 'lstm' ) ];
            end
            
            layersEnc = [ layersEnc; ...
                layerNormalizationLayer( 'Name', 'lnorm' )
                leakyReluLayer( self.scale, 'Name', 'relu' )
                spatialDropoutLayer( self.dropout, 'Name', 'drop1' )
                fullyConnectedLayer( self.ZDim, 'Name', 'out' )
                ];

            lgraphEnc = layerGraph( layersEnc );
                   
            self.nets.encoder = dlnetwork( lgraphEnc );

        end


        function self = initDecoder( self )
            % Initialize the decoder network
            arguments
                self        lstmModel
            end

            layersDec = [ ...
                sequenceInputLayer( self.XChannels, 'Name', 'in' )
                ];

            if self.bidirectional
                layersDec = [ layersDec; ...
                    bilstmLayer( self.ZDim, 'OutputMode', 'last', ...
                                            'HasStateInputs', true, ...
                                            'HasStateOutputs', true, ...
                                            'Name', 'lstm' ) ];
            else
                layersDec = [ layersDec; ...
                    lstmLayer( self.ZDim, 'OutputMode', 'last', ...
                                          'HasStateInputs', true, ...
                                          'HasStateOutputs', true, ...
                                          'Name', 'lstm' ) ];
            end

            layersDec = [ layersDec; ...
                layerNormalizationLayer( 'Name', 'lnorm' )
                leakyReluLayer( self.scale, 'Name', 'relu' )
                spatialDropoutLayer( self.dropout, 'Name', 'drop1' )
                fullyConnectedLayer( self.XChannels, 'Name', 'out' )
                ];

            lgraphDec = layerGraph( layersDec );

            % insert extra inputs
            lgraphDec = addLayers( lgraphDec, ...
                           featureInputLayer( self.ZDim, 'Name', 'hidden' ) );
            lgraphDec = addLayers( lgraphDec, ...
                           featureInputLayer( self.ZDim, 'Name', 'cell' ) );
            lgraphDec = connectLayers( lgraphDec, 'hidden', 'lstm/hidden' );
            lgraphDec = connectLayers( lgraphDec, 'cell', 'lstm/cell' );
            
            self.nets.decoder = dlnetwork( lgraphDec );

        end


        function [ dlXHat, dlZ, state ] = forward( self, encoder, decoder, dlX )
            % Forward-run the lstm network, overriding autoencoder method
            arguments
                self        lstmModel
                encoder     dlnetwork
                decoder     dlnetwork
                dlX         dlarray
            end

            % generate latent encodings
            [ dlZ, state.encoder ] = forward( encoder, dlX );

            % initialize the hidden states (HS) and cell states (CS)
            dlHS = dlZ;
            dlCS = dlarray( zeros(size(dlZ), 'like', dlZ), 'CB' );
            
            if self.bidirectional
                dlHS = repmat( dlHS, [2 1] );
                dlCS = repmat( dlCS, [2 1]);
            end

            % reconstruct curves using teacher forcing/free running
            if self.reverseDecoding
                dlX = flip( dlX, 3 );
            end
            seqLen = size( dlX, 3 );

            dlXHat = cast( zeros( size(dlX,1), size(dlX,2), seqLen+1 ), ...
                           'like', dlX );
            
            if self.scheduleSampling
                rate = min( self.trainer.currentEpoch...
                                *self.samplingRateIncrement, 1 );
                mask = rand( seqLen, 1 ) < rate;
            end
            
            for i = 1:seqLen

                if self.scheduleSampling && mask(i) 
                    % free running: last prediction
                    dlNextX = dlXHat(:,:,i);
                else
                    % teacher forcing: ground truth
                    dlNextX = dlX(:,:,i);
                end
            
                [ dlCS, dlHS, dlXHat(:,:,i+1), state.dec ] = ...
                                  forward( decoder, dlNextX, dlHS, dlCS );
                
                decoder.State = state.dec;

            end

            % align sequences correctly
            dlXHat = dlXHat(:,:,2:end);
            if self.reverseDecoding
                dlXHat = flip( dlXHat, 3 );
            end


        end

    end

end


