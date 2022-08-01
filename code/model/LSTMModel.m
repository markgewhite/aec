classdef LSTMModel < FCModel
    % Subclass defining a long/short term memory autoencoder model

    properties
        NumHiddenUnits          % number LSTM nodes
        ReverseDecoding         % whether to reverse the order when decoding
        Bidirectional           % if the network is bidirectional
        ScheduleSampling        % whether to perform schedule sampling
        SamplingRateIncrement   % sampling rate increment per epoch
        HasFCDecoder            % whether the decoder inherits the fully-connected design
    end

    methods

        function self = LSTMModel( thisDataset, ...
                                   lossFcns, ...
                                   superArgs, ...
                                   superArgs2, ...
                                   args )
            % Initialize the model
            arguments
                thisDataset             ModelDataset
            end
            arguments (Repeating)
                lossFcns                LossFunction
            end
            arguments
                superArgs.?FCModel
                superArgs2.name         string
                superArgs2.path         string
                args.NumHiddenUnits     double ...
                    {mustBeInteger, mustBePositive} = 50
                args.ReverseDecoding    logical = true
                args.Bidirectional      logical = false
                args.ScheduleSampling   logical = true
                args.SamplingRateIncrement double ...
                    {mustBePositive} = 0.005
                args.HasFCDecoder       logical = false
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            superArgs2Cell = namedargs2cell( superArgs2 );

            self@FCModel( thisDataset, ...
                          lossFcns{:}, ...
                          superArgsCell{:}, ...
                          superArgs2Cell{:}, ...
                          FlattenInput = false, ...
                          HasSeqInput = true, ...
                          IsVAE = false );

            % store this class's properties
            self.NumHiddenUnits = args.NumHiddenUnits;
            self.ReverseDecoding = args.ReverseDecoding;
            self.Bidirectional = args.Bidirectional;
            self.ScheduleSampling = args.ScheduleSampling;
            self.SamplingRateIncrement = args.SamplingRateIncrement;
            self.HasFCDecoder = args.HasFCDecoder;

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

            if self.HasFCDecoder
                net = initDecoder@FCModel( self );
                return
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


        function self = initSubModel( self, k )
            % Initialize a sub-model
            arguments
                self            LSTMModel
                k               double
            end

            self.SubModels{k} = LSTMCompactModel( self, k );
            if self.IdenticalNetInit && k==1
                self.InitializedNets = self.SubModels{k}.Nets;
            end

        end


        function dlZ = predict( encoder, X, arg )
            % Override the predict function of a fully-connected network
            arguments
                encoder         dlnetwork
                X               {mustBeA( X, {'dlarray', 'ModelDataset'} )}
                arg.convert     logical = true
            end

            dlZ = predict@FullAEModel( encoder, X, arg );

        end

    end

end


