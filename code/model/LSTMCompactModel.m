classdef LSTMCompactModel < CompactAEModel
    % Subclass to define LSTM-specific forward and encode functions

    properties
        ReverseDecoding         % whether to reverse the order when decoding
        Bidirectional           % if the network is bidirectional
        ScheduleSampling        % whether to perform schedule sampling
        SamplingRateIncrement   % sampling rate increment per epoch
    end

    methods

        function self = LSTMCompactModel( theLSTMModel, fold )
            % Initialize the model
            arguments
                theLSTMModel           LSTMModel
                fold                   double
            end

            self@CompactAEModel( theLSTMModel, fold );

            self.ReverseDecoding = theLSTMModel.ReverseDecoding;
            self.Bidirectional = theLSTMModel.Bidirectional;
            self.ScheduleSampling = theLSTMModel.ScheduleSampling;
            self.SamplingRateIncrement = theLSTMModel.SamplingRateIncrement;

        end


        function [ dlXHat, dlZ, state ] = forward( self, encoder, decoder, dlX )
            % Forward-run the lstm network, overriding autoencoder method
            arguments
                self        LSTMCompactModel
                encoder     dlnetwork
                decoder     dlnetwork
                dlX         dlarray
            end

            % generate latent encodings
            [ dlZ, state.Encoder ] = forward( encoder, dlX );

            % reconstruct curves from latent codes
            [ dlXHat, state.Decoder ] = self.forwardDecoder( decoder, dlZ, dlX );

        end


        function dlX = decodeDispatcher( self, dlZ, args )
            % Generate X from Z either using forward or predict
            % Overrides the autoencoder method
            arguments
                self                LSTMCompactModel
                dlZ                 dlarray
                args.forward        logical = false
                args.dlX            dlarray
            end

            if args.forward
                dlX = self.forwardDecoder( self.Nets.Decoder, ...
                                       dlZ, args.dlX );
            else
                dlX = self.predictDecoder( self.Nets.Decoder, dlZ );
            end

        end


        function dlZ = encode( self, X, arg )
            % Encode features Z from X using the model
            % overriding the autoencoder encode method
            % which must have inputs in the trained batch size
            arguments
                self            LSTMCompactModel
                X               
                arg.convert     logical = true
            end

            if isa( X, 'ModelDataset' )
                dlX = X.getDLInput( self.XDimLabels );
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


    methods (Access = private)

        function [ dlXHat, state ] = forwardDecoder( self, decoder, dlHS, dlX )
            % Forward-run the lstm decoder network
            arguments
                self            LSTMCompactModel
                decoder         dlnetwork
                dlHS            dlarray
                dlX             dlarray
            end
        
            % initialize the hidden states (HS) and cell states (CS)
            dlCS = dlarray( zeros(size(dlHS), 'like', dlHS), 'CB' );
            
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

                [ dlCS, dlHS, dlXHat(:,:,i), state ] = ...
                                  forward( decoder, dlNextX, dlHS, dlCS );

                decoder.State = state;
        
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


        function dlXHat = predictDecoder( self, decoder, dlHS )
            % Forward-run the lstm decoder network
            arguments
                self            LSTMCompactModel
                decoder         dlnetwork
                dlHS            dlarray
            end
        
            % initialize the hidden states (HS) and cell states (CS)
            dlCS = dlarray( zeros(size(dlHS), 'like', dlHS), 'CB' );
            
            if self.Bidirectional
                dlHS = repmat( dlHS, [2 1] );
                dlCS = repmat( dlCS, [2 1]);
            end
               
            seqOutputLen = self.XTargetDim;
            dlNextX = dlarray( zeros( self.XChannels, size(dlHS,2) ), 'CBT' );
            dlXHat = dlarray( zeros( self.XChannels, size(dlHS,2), ...
                                     seqOutputLen), 'CBT' );
            
            for i = 1:seqOutputLen
        
                [ dlCS, dlHS, dlXHat(:,:,i) ] = ...
                                  forward( decoder, dlNextX, dlHS, dlCS );
                
                dlNextX = dlXHat(:,:,i);

            end
        
            % align sequences correctly
            dlXHat = dlXHat(:,:,1:seqOutputLen);
            
            % permute to match dlXOut
            % (tracing will be taken care of in recon loss calculation)
            dlXHat = double(extractdata( dlXHat ));
            dlXHat = permute( dlXHat, [3 2 1] );
        
        end


    end


    

end