classdef FCBranchedModel < BranchedModel
    % Subclass defining a fully connected autoencoder model
    properties
        NumFC                 % number of nodes for widest layer
        FCFactor              % log2 scaling factor subsequent layers
        NumFCDecoder          % number of nodes for widest layer
        FCFactorDecoder       % log2 scaling factor subsequent layers
    end

    methods

        function self = FCBranchedModel( thisDataset, ...
                                 superArgs, ...
                                 superArgs2, ...
                                 args )
            % Initialize the model
            arguments
                thisDataset     ModelDataset
                superArgs.?BranchedModel
                superArgs2.name     string
                superArgs2.path     string
                args.NumHidden      double ...
                    {mustBeInteger, mustBePositive} = 2
                args.NumFC          double ...
                    {mustBeInteger, mustBePositive} = 64
                args.FCFactor       double ...
                    {mustBeInteger, mustBePositive} = 2
                args.NumFCDecoder       double ...
                    {mustBeInteger, mustBePositive} = 10
                args.FCFactorDecoder    double ...
                    {mustBeInteger} = 0
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            superArgs2Cell = namedargs2cell( superArgs2 );

            self@BranchedModel( thisDataset, ...
                          superArgsCell{:}, ...
                          superArgs2Cell{:} );

            % store this class's properties
            self.NumFC = args.NumFC;
            self.FCFactor = args.FCFactor;
            self.NumFCDecoder = args.NumFCDecoder;
            self.FCFactorDecoder = args.FCFactorDecoder;
        end


        function [lgraph, lastLayerName] = initEncoderInputLayers( self )
            % Initialize the encoder's input layers
             arguments
                self        FCBranchedModel       
             end
            
             if self.HasInputNormalization
                layers = featureInputLayer( self.XInputDim*self.XChannels, ...
                                            'Name', 'in', ...
                                            'Normalization', 'zscore', ...
                                            'Mean', 0, 'StandardDeviation', 1 );
            else
                layers = featureInputLayer( self.XInputDim*self.XChannels, ...
                                            'Name', 'in' );
            end

            if self.InputDropout > 0
                layers = [ layers; ...
                           dropoutLayer( self.InputDropout, 'Name', 'drop0' ) ];
                lastLayerName = 'drop0';
            else
                lastLayerName = 'in';
            end

            lgraph = layerGraph( layers );

        end


        function [lgraph, finalLayerName] = initEncoderHiddenLayers( self, ...
                                                     lgraph, lastLayerName, offset )
            % Initialize the encoder's hidden layers
            arguments
                self            FCBranchedModel
                lgraph   
                lastLayerName   char
                offset          double = 0
            end
            
            for i = 1:self.NumHidden

                nNodes = fix( self.NumFC*2^(self.FCFactor*(1-i)) );
                if nNodes < self.ZDim
                    eid = 'FCModel:Design';
                    msg = 'Encoder hidden layer smaller than latent space.';
                    throwAsCaller( MException(eid,msg) );
                end

                fcLayer = fullyConnectedLayer( nNodes, ...
                                               'Name', ['fc' num2str(i+offset)] );

                [lgraph, lastLayerName] = self.addBlock( fcLayer, ...
                                    lgraph, i + offset, lastLayerName, ...
                                    self.ReluScale, ...
                                    self.Dropout, ...
                                    self.NetNormalizationType, ...
                                    self.NetActivationType );

            end

            % add the final layer with double the dimensions for VAE
            finalLayerName = ['fcout' num2str(100*offset)];
            lgraph = addLayers( lgraph, ...
                                fullyConnectedLayer( self.ZDim*2, 'Name', ...
                                                     finalLayerName ) );
            lgraph = connectLayers( lgraph, lastLayerName, finalLayerName );
            
        end


        function [lgraph, lastLayerName] = initDecoderInputLayers( self )
            % Initialize the decoder's input layers
            arguments
                self        FCBranchedModel
            end

            inLayer = featureInputLayer( self.ZDim, 'Name', 'in' );
            lgraph = layerGraph( inLayer );

            lastLayerName = 'in';

        end



        function [lgraph, finalLayerName] = initDecoderHiddenLayers( self, ...
                                                     lgraph, lastLayerName, offset )
            % Initialize the decoder's hidden layers
            arguments
                self            FCBranchedModel
                lgraph   
                lastLayerName   char
                offset          double = 0
            end

            for i = 1:self.NumHiddenDecoder

                nNodes = fix( self.NumFCDecoder*2^(self.FCFactor*(-self.NumHiddenDecoder+i)) );
                if nNodes < self.ZDim
                    eid = 'FCModel:Design';
                    msg = 'Decoder hidden layer smaller than latent space.';
                    throwAsCaller( MException(eid,msg) );
                end

                fcLayer = fullyConnectedLayer( nNodes, ...
                                               'Name', ['fc' num2str(i+offset)] );

                [lgraph, lastLayerName] = self.addBlock( fcLayer, ...
                                    lgraph, i + offset, lastLayerName, ...
                                    self.ReluScaleDecoder, ...
                                    self.DropoutDecoder, ...
                                    self.NetNormalizationTypeDecoder, ...
                                    self.NetActivationTypeDecoder );

            end

            finalLayerName = ['comp' num2str(offset)];
            lgraph = addLayers( lgraph, ...
                                fullyConnectedLayer( self.XTargetDim*self.XChannels, ...
                                                     'Name', finalLayerName ) );

            lgraph = connectLayers( lgraph, lastLayerName, finalLayerName );

        end


        function self = setXTargetDim( self )
            % Calculate the decoder's output size
            arguments
                self           FCBranchedModel
            end

            self.XTargetDim = self.NumFCDecoder;

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
                dlX = self.getDLArrays( X );
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


