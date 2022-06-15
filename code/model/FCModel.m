classdef FCModel < FullAEModel
    % Subclass defining a fully connected autoencoder model
    properties
        NumHidden     % number of hidden layers
        NumFC         % number of nodes for widest layer
        FCFactor      % log2 scaling factor subsequent layers
        ReLuScale     % leaky ReLu scale factor
        InputDropout  % input dropout rate
        Dropout       % hidden layer dropout rate

    end

    methods

        function self = FCModel( thisDataset, ...
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
                superArgs2.name     string
                superArgs2.path     string
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
            superArgs2Cell = namedargs2cell( superArgs2 );

            self@FullAEModel( thisDataset, ...
                              lossFcns{:}, ...
                              superArgsCell{:}, ...
                              superArgs2Cell{:}, ...
                              flattenInput = true, ...
                              hasSeqInput = false );

            % store this class's properties
            self.NumHidden = args.nHidden;
            self.NumFC = args.nFC;
            self.FCFactor = args.fcFactor;
            self.ReLuScale = args.scale;
            self.InputDropout = args.inputDropout;
            self.Dropout = args.dropout;
           
        end


        function net = initEncoder( self )
            % Initialize the encoder network
            arguments
                self        FCModel
            end

            layersEnc = [ featureInputLayer( self.XInputDim*self.XChannels, ...
                                   'Name', 'in', ...
                                   'Normalization', 'zscore', ...
                                   'Mean', 0, 'StandardDeviation', 1 )
                          dropoutLayer( self.InputDropout, 'Name', 'drop0' ) ];

            lgraphEnc = layerGraph( layersEnc );       
            lastLayer = 'drop0';
            
            for i = 1:self.NumHidden

                nNodes = fix( self.NumFC*2^(self.FCFactor*(1-i)) );

                [lgraphEnc, lastLayer] = addBlock( lgraphEnc, i, lastLayer, ...
                                    nNodes, self.ReLuScale, self.Dropout );

            end
            
            outLayers = fullyConnectedLayer( self.ZDim*(self.IsVAE+1), ...
                                               'Name', 'out' );
            
            lgraphEnc = addLayers( lgraphEnc, outLayers );
            lgraphEnc = connectLayers( lgraphEnc, ...
                                       lastLayer, 'out' );

            if self.IsVAE
                net = VAEdlnetwork( lgraphEnc, numDraws = self.NumVAEDraws );
            else
                net = dlnetwork( lgraphEnc );
            end

        end


        function net = initDecoder( self )
            % Initialize the decoder network
            arguments
                self        FCModel
            end

            layersDec = featureInputLayer( self.ZDim, 'Name', 'in' );
            
            lgraphDec = layerGraph( layersDec );
            lastLayer = 'in';
            
            for i = 1:self.NumHidden

                nNodes = fix( self.NumFC*2^(self.FCFactor*(-self.NumHidden+i)) );

                [lgraphDec, lastLayer] = addBlock( lgraphDec, i, lastLayer, ...
                                    nNodes, self.ReLuScale, self.Dropout );

            end

            outLayers = fullyConnectedLayer( self.XTargetDim*self.XChannels, ...
                                               'Name', 'fcout' );

            if self.XChannels > 1
                outLayers = [ outLayers; 
                                reshapeLayer( [self.XTargetDim self.XChannels], ...
                                              'Name', 'reshape' ) ];
            end
            
            lgraphDec = addLayers( lgraphDec, outLayers );
            lgraphDec = connectLayers( lgraphDec, ...
                                       lastLayer, 'fcout' );

            net = dlnetwork( lgraphDec );

        end


        function dlZ = predict( encoder, X, arg )
            % Override the predict function for a fully-connected network
            % to flatten the input array
            arguments
                encoder         dlnetwork
                X               {mustBeA( X, {'dlarray', 'ModelDataset'} )}
                arg.convert     logical = true
            end

            if isa( X, 'ModelDataset' )
                dlX = X.getDLInput( self.XDimLabels );
            else
                dlX = X;
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
            dlZ = predict( encoder, dlX );

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