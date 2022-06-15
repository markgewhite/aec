classdef LSTMModel < FullAEModel
    % Subclass defining a long/short term memory autoencoder model

    properties
        NumHiddenUnits          % number LSTM nodes
        ReLuScale               % leaky ReLu scale factor
        InputDropout            % initial dropout rate
        Dropout                 % dropout rate
        ReverseDecoding         % whether to reverse the order when decoding
        Bidirectional           % if the network is bidirectional
        ScheduleSampling        % whether to perform schedule sampling
        SamplingRateIncrement   % sampling rate increment per epoch
    end

    methods

        function self = LSTMModel( thisDataset, ...
                                   lossFcns, ...
                                   superArgs, ...
                                   superArgs2, ...
                                   args )
            % Initialize the model
            arguments
                thisDataset     ModelDataset
            end
            arguments (Repeating)
                lossFcns        LossFunction
            end
            arguments
                superArgs.?FullAEModel
                superArgs2.name         string
                superArgs2.path         string
                args.numHiddenUnits     double ...
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
            superArgs2Cell = namedargs2cell( superArgs2 );

            self@FullAEModel( thisDataset, ...
                              lossFcns{:}, ...
                              superArgsCell{:}, ...
                              superArgs2Cell{:}, ...
                              hasSeqInput = true, ...
                              isVAE = false );

            % store this class's properties
            self.NumHiddenUnits = args.numHiddenUnits;
            self.ReLuScale = args.scale;
            self.InputDropout = args.inputDropout;
            self.Dropout = args.dropout;

            self.ReverseDecoding = args.reverseDecoding;
            self.Bidirectional = args.bidirectional;
            self.ScheduleSampling = args.scheduleSampling;
            self.SamplingRateIncrement = args.samplingRateIncrement;

        end


        function net = initEncoder( self )
            % Initialize the encoder network
            arguments
                self        LSTMModel
            end

            layersEnc = [ ...
                sequenceInputLayer( self.XChannels, 'Name', 'in', ...
                                   'Normalization', 'zscore', ...
                                   'Mean', 0, 'StandardDeviation', 1 )
                dropoutLayer( self.InputDropout, 'Name', 'drop0' )
                ];
            
            if self.Bidirectional
                layersEnc = [ layersEnc; ...
                    bilstmLayer( self.NumHiddenUnits, ...
                                    'OutputMode', 'last', ...
                                     'Name', 'lstm' ) ];
            else
                layersEnc = [ layersEnc; ...
                    lstmLayer( self.NumHiddenUnits, ...
                                    'OutputMode', 'last', ...
                                     'Name', 'lstm' ) ];
            end
            
            layersEnc = [ layersEnc; ...
                fullyConnectedLayer( self.ZDim, 'Name', 'out' )
                ];

            lgraphEnc = layerGraph( layersEnc );
                   
            net = dlnetwork( lgraphEnc );

        end


        function net = initDecoder( self )
            % Initialize the decoder network
            arguments
                self        LSTMModel
            end

            layersDec = [ ...
                sequenceInputLayer( self.XChannels, 'Name', 'in' )
                ];

            if self.Bidirectional
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
            
            net = dlnetwork( lgraphDec );

        end


        function [ dlXHat, dlZ, state ] = forward( self, encoder, decoder, dlX )
            % Forward-run the lstm network, overriding autoencoder method
            arguments
                self        LSTMModel
                encoder     dlnetwork
                decoder     dlnetwork
                dlX         dlarray
            end

            % generate latent encodings
            [ dlZ, state.Encoder ] = forward( encoder, dlX );

            % initialize the hidden states (HS) and cell states (CS)
            dlHS = dlZ;
            dlCS = dlarray( zeros(size(dlZ), 'like', dlZ), 'CB' );
            
            if self.Bidirectional
                dlHS = repmat( dlHS, [2 1] );
                dlCS = repmat( dlCS, [2 1]);
            end

            % reconstruct curves using teacher forcing/free running
            if self.ReverseDecoding
                dlX = flip( dlX, 3 );
            end

            seqInputLen = size( dlX, 3 );
            seqOutputLen = self.XTargetDim;

            dlXHat = repmat( dlX, [1 1 2] );
            
            if self.ScheduleSampling
                rate = min( self.Trainer.CurrentEpoch...
                                *self.SamplingRateIncrement, 1 );
                mask = rand( seqOutputLen, 1 ) < rate;
            end
            
            for i = 1:seqOutputLen

                if i > 1
                    if (self.ScheduleSampling && mask(i)) ...
                            || i > seqInputLen
                        % free running: last prediction
                        dlNextX = dlXHat(:,:,i-1);
                    else
                        % teacher forcing: ground truth
                        dlNextX = dlX(:,:,i-1);
                    end
                else
                    dlNextX = 0*dlX(:,:,1);
                end

                [ dlCS, dlHS, dlXHat(:,:,i), state.dec ] = ...
                                  forward( decoder, dlNextX, dlHS, dlCS );
                
                decoder.State = state.dec;

            end

            % align sequences correctly
            dlXHat = dlXHat(:,:,1:seqOutputLen);
            %if self.reverseDecoding
            %    dlXHat = flip( dlXHat, 3 );
            %end
            
            % permute to match dlXOut
            % (tracing will be taken care of in recon loss calculation)
            dlXHat = double(extractdata( dlXHat ));
            dlXHat = permute( dlXHat, [3 2 1] );

        end

    end

    methods (Static)

        function dlZ = encode( self, X, arg )
            % Encode features Z from X using the model
            % overriding the autoencoder encode method
            % which must have inputs in the trained batch size
            arguments
                self            autoencoderModel
                X
                arg.convert     logical = true
            end

            if isa( X, 'ModelDataset' )
                dlX = X.getDLInput;
            elseif isa( X, 'dlarray' )
                dlX = X;
            else
                eid = 'Autoencoder:NotValidX';
                msg = 'The input data should be a ModelDataset or a dlarray.';
                throwAsCaller( MException(eid,msg) );
            end

            batchSize = size( self.Nets.Encoder.State.Value{1}, 2 );
            nObs = size( dlX, 2 );
            nBatches = fix(nObs/batchSize);
            dlZ = dlarray( zeros( self.ZDim, nObs), 'CB' );

            j = 1;
            % make predictions in batches
            for i = 1:nBatches
                dlZ(:,j:j+batchSize-1) = predict( self.Nets.Encoder, ...
                                dlX( :, j:j+batchSize-1, :)  );
                j = j+batchSize;
            end
            % cover the remainder
            dlZ( :, end-batchSize+1:end ) = predict( self.Nets.Encoder, ...
                                dlX( :, end-batchSize+1:end, : ) );

            if arg.convert
                dlZ = double(extractdata( dlZ ))';
            end

        end

    end

end


