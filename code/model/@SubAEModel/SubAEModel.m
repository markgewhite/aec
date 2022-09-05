classdef SubAEModel < SubRepresentationModel
    % Subclass defining the framework for an autoencoder model
    
    properties
        Nets           % networks defined in this model (structure)
        NetNames       % names of the networks (for convenience)
        NumNetworks    % number of networks
        LossFcns       % array of loss functions
        LossFcnNames   % names of the loss functions
        LossFcnWeights % weights to be applied to the loss function
        LossFcnTbl     % convenient table summarising loss function details
        NumLoss        % number of computed losses
        FlattenInput   % whether to flatten input
        HasSeqInput    % supports variable-length input
        Trainer        % trainer object holding training parameters
        Optimizer      % optimizer object
        ZDimActive     % number of dimensions currently active
        ComponentCentring % how to centre the generated components
        HasCentredDecoder % whether the decoder predicts centred X
        MeanCurveTarget   % mean curve for the X target time span
        AuxNetworkALE  % auxiliary network Accumulated Local Effects
    end

    properties (Dependent = true)
        XDimLabels     % dimensional labelling for X input dlarrays
        XNDimLabels    % dimensional labelling for time-normalized output
    end

    methods

        function self = SubAEModel( theFullModel, fold )
            % Initialize the model
            arguments
                theFullModel        FullAEModel
                fold                double
            end

            self@SubRepresentationModel( theFullModel, fold );

            % copy over the full model's relevant properties
            self.NetNames = theFullModel.NetNames;
            self.NumNetworks = theFullModel.NumNetworks;
            self.LossFcns = theFullModel.LossFcns;
            self.LossFcnNames = theFullModel.LossFcnNames;
            self.LossFcnWeights = theFullModel.LossFcnWeights;
            self.LossFcnTbl = theFullModel.LossFcnTbl;
            self.NumLoss = theFullModel.NumLoss;
            self.FlattenInput = theFullModel.FlattenInput;
            self.HasSeqInput = theFullModel.HasSeqInput;
            self.ZDimActive = theFullModel.InitZDimActive;
            self.ComponentCentring = theFullModel.ComponentCentring;
            self.HasCentredDecoder = theFullModel.HasCentredDecoder;

            if theFullModel.IdenticalNetInit ...
                    && ~isempty( theFullModel.InitializedNets )
                % use the common network initializtion
                self.Nets = theFullModel.InitializedNets;

            else
                % perform a network initialization unique to this object
                % initialize the encoder and decoder networks
                self.Nets.Encoder = theFullModel.initEncoder;
                self.Nets.Decoder = theFullModel.initDecoder;
    
                % initialize the loss function networks, if required
                self = self.initLossFcnNetworks;
            end

            % initialize the trainer
            try
                argsCell = namedargs2cell( theFullModel.Trainer );
            catch
                argsCell = {};
            end
            self.Trainer = ModelTrainer( self.LossFcnTbl, ...
                                         argsCell{:}, ...
                                         showPlots = self.ShowPlots );

            % initialize the optimizer
            try
                argsCell = namedargs2cell( theFullModel.Optimizer );
            catch
                argsCell = {};
            end
            self.Optimizer = ModelOptimizer( self.NetNames, argsCell{:} );

        end


        function labels = get.XDimLabels( self )
            % Get the X dimensional labels for dlarrays
            arguments
                self            SubAEModel
            end

            if self.XChannels==1
                if self.HasSeqInput
                    labels = 'TBC';
                else
                    labels = 'CB';
                end
            else
                if self.HasSeqInput
                    labels = 'TBC';
                else
                    labels = 'SBC';
                end
            end
            
        end


        function labels = get.XNDimLabels( self )
            % Get the XN dimensional labels for dlarrays
            arguments
                self            SubAEModel
            end

            if self.XChannels==1
                labels = 'CB';
            else
                labels = 'SBC';
            end
            
        end

        % class methods

        [ XC, XMean, offsets ] = calcLatentComponents( self, dlZ, args )

        self = compress( self, level )

        dlX = decodeDispatcher( self, dlZ, args )

        dlZ = encode( self, X, arg )

        [ dlZ, dlXHat, state ] = forward( self, encoder, decoder, dlX )

        [ dlXHat, state ] = forwardDecoder( self, decoder, dlZ )

        [ dlZ, state ] = forwardEncoder( self, encoder, dlX )

        self = incrementActiveZDim( self )

        self = initLossFcnNetworks( self )

        dlZ = maskZ( self, dlZ )

        self = train( self, thisData )

        [ YHat, YHatScore] = predictAuxNet( self, Z )

        YHat = predictCompNet( self, thisDataset )

        [ dlXHat, XHatSmth, XHatReg ] = reconstruct( self, Z, args )

    end


    methods (Static)

        [ eval, pred, cor ] = evaluateSet( self, thisDataset )

    end

end