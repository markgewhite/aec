classdef CompactAEModel < CompactRepresentationModel
    % Subclass defining the framework for an autoencoder model
    
    properties
        Nets           % networks defined in this model (structure)
        NetNames       % names of the networks (for convenience)
        NumNetworks    % number of networks
        IsVAE          % flag indicating if variational autoencoder
        NumVAEDraws    % number of draws from encoder output distribution
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

        function self = CompactAEModel( theFullModel, fold )
            % Initialize the model
            arguments
                theFullModel        FullAEModel
                fold                double
            end

            self@CompactRepresentationModel( theFullModel, fold );

            % copy over the full model's relevant properties
            self.NetNames = theFullModel.NetNames;
            self.NumNetworks = theFullModel.NumNetworks;
            self.IsVAE = theFullModel.IsVAE;
            self.NumVAEDraws = theFullModel.NumVAEDraws;
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


        function self = initLossFcnNetworks( self )
            % Initialize the loss function networks
            arguments
                self        CompactAEModel
            end

            for i = 1:length( self.LossFcnNames )
                thisName = self.LossFcnNames{i};
                thisLossFcn = self.LossFcns.(thisName);
                thisType = self.LossFcnTbl.Types(self.LossFcnTbl.Names == thisName);
                if thisLossFcn.HasNetwork
                    if thisType == 'Comparator' %#ok<BDSCA> 
                        self.Nets.(thisName) = thisLossFcn.initNetwork( ...
                                                self.Nets.Encoder );
                    else
                        self.Nets.(thisName) = thisLossFcn.initNetwork;
                    end
                end
            end

        end


        function self = train( self, thisData )
            % Train the autoencoder
            arguments
                self            CompactAEModel
                thisData        ModelDataset
            end

            self = self.Trainer.runTraining( self, thisData );
            
            [self.AuxModelALE, self.ALEQuantiles, ...
                self.LatentComponents, ...
                self.VarProportion, self.ComponentVar] ...
                            = self.getLatentResponse( thisData );

            if any(self.LossFcnTbl.Types=='Auxiliary')
                self.AuxNetworkALE = ...
                    self.auxPartialDependence( thisData, ...
                                               auxFcn = @predictAuxNet );
            end

        end


        function labels = get.XDimLabels( self )
            % Get the X dimensional labels for dlarrays
            arguments
                self            CompactAEModel
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
                self            CompactAEModel
            end

            if self.XChannels==1
                labels = 'CB';
            else
                labels = 'SBC';
            end
            
        end


        function [ XC, XMean, offsets ] = calcLatentComponents( self, dlZ, args )
            % Calculate the funtional components using the decoder network
            % from Z code generated using Accumulated Local Estimation 
            arguments
                self            CompactAEModel
                dlZ             {mustBeA( dlZ, {'dlarray', 'double'} )}
                args.maxObs     double {mustBeInteger} = 500
                args.forward    logical = false
                args.smooth     logical = false
                args.dlX        {mustBeA( args.dlX, {'dlarray', 'double'} )}
            end

            % prepare decoder dispatch arguments
            dispatchArgs.forward = args.forward;
            if isfield( args, 'dlX' )
                dispatchArgs.dlX = args.dlX;
            end
            dispatchArgsCell = namedargs2cell( dispatchArgs );

            [XC, ~, ~, offsets] = self.ALE( dlZ, ...
                              sampling = 'Component', ...
                              modelFcn = @decodeDispatcher, ...
                              modelFcnArgs = dispatchArgsCell, ...
                              maxObs = args.maxObs );

            % put XC into the appropriate structure
            % Points, Samples, Components, Channels
            XC = permute( XC, [3 2 1 4] );
            nSamples = size(XC, 2);

            % extract the mean curve based on Z
            XMean = XC( :, ceil(nSamples/2), :, : );
            
            switch self.ComponentCentring
                case 'Z'
                    % centre about the curve generated by mean Z
                    XC = XC - XMean;
                case 'X'
                    % centre about the mean generated curve
                    XC = XC - mean( XC, length(size(XC)) );
            end

            if args.smooth
                % smooth to regularly-spaced time span
                XCSmth = zeros( length(self.TSpan.Regular), ...
                                size(XC,2), size(XC,3), size(XC,4) );
                for c = 1:size(XC,4)
                    XCSmth(:,:,:,c) = smoothSeries( XC(:,:,:,c), ...
                                         self.TSpan.Target, ...
                                         self.TSpan.Regular, ...
                                         self.FDA.FdParamsTarget );
                end
                XC = XCSmth;
            end

        end


        function [ dlZ, dlXHat, state, dlMean, dlLogVar ] = ...
                                    forward( self, encoder, decoder, dlX )
            % Forward-run the encoder and decoder networks
            arguments
                self        CompactAEModel
                encoder     dlnetwork
                decoder     dlnetwork
                dlX         dlarray
            end

            % generate latent encodings
            [ dlZ, state.Encoder, dlMean, dlLogVar ] = ...
                                        self.forwardEncoder( encoder, dlX );

            % reconstruct curves from latent codes
            [ dlXHat, state.Decoder ] = self.forwardDecoder( decoder, dlZ );

        end


        function [ dlZ, state, dlMean, dlLogVar ] = forwardEncoder( self, encoder, dlX )
            % Forward-run the encoder network
            % dlnetworks are provided for tracing purposes 
            % rather than using the object's network definitions
            arguments
                self        CompactAEModel
                encoder     dlnetwork
                dlX         dlarray
            end

            if self.FlattenInput && size( dlX, 3 ) > 1
                dlX = flattenDLArray( dlX );
            end

            % generate latent encodings
            [ dlEncOutput, state ] = forward( encoder, dlX );

            if self.IsVAE
                % variational autoencoder
                [ dlZ, dlMean, dlLogVar ] = ...
                    reparameterize( dlEncOutput, self.NumVAEDraws );
            else
                % standard encoder
                dlZ = dlEncOutput;
                dlMean = mean( dlZ, 2 );
                dlLogVar = log( var( dlZ, [], 2 ) );
            end
    
            % mask Z based on number of active dimensions
            dlZ = self.maskZ( dlZ );

        end


        function [ dlXHat, state ] = forwardDecoder( self, decoder, dlZ )
            % Forward-run the decoder network
            % dlnetworks are provided for tracing purposes 
            % rather than using the object's network definitions
            arguments
                self        CompactAEModel
                decoder     dlnetwork
                dlZ         dlarray
            end

            % reconstruct curves from latent codes
            [ dlXHat, state ] = forward( decoder, dlZ );

        end


        function dlZ = encode( self, X, arg )
            % Encode features Z from X using the model
            arguments
                self            CompactAEModel
                X               {mustBeA(X, {'ModelDataset', 'dlarray'})}
                arg.convert     logical = true
            end

            if isa( X, 'ModelDataset' )
                dlX = X.getDLInput( self.XDimLabels );
            else
                dlX = X;
            end

            if self.FlattenInput && size( dlX, 3 ) > 1
                dlX = flattenDLArray( dlX );
            end

            dlEncOutput = predict( self.Nets.Encoder, dlX );

            if self.IsVAE
                dlZ = dlEncOutput( 1:self.ZDim, : );
            else
                dlZ = dlEncOutput;
            end
            
            if arg.convert
                dlZ = double(extractdata( dlZ ))';
            end

        end


        function [ dlXHat, XHatSmth, XHatReg ] = reconstruct( self, Z, args )
            % Reconstruct X from Z using the model
            arguments
                self            CompactAEModel
                Z               {mustBeA(Z, {'double', 'dlarray'})}
                args.convert    logical = true
            end

            if isa( Z, 'dlarray' )
                dlZ = Z;
            else
                dlZ = dlarray( Z', 'CB' );
            end

            dlXHat = decodeDispatcher( self, dlZ, forward = false );

            if self.HasCentredDecoder
                dlXHat = dlXHat + self.MeanCurveTarget;
            end

            XHatSmth = [];
            XHatReg = [];

            if args.convert

                if isa( dlXHat, 'dlarray' )
                    dlXHat = double(extractdata( dlXHat ));
                end
                dlXHat = permute( dlXHat, [1 3 2] );
                dlXHat = squeeze( dlXHat ); 

                XHatSmth = smoothSeries(  dlXHat, ...
                                          self.TSpan.Target, ...
                                          self.TSpan.Target, ...
                                          self.FDA.FdParamsTarget );
                XHatReg = smoothSeries(   dlXHat, ...
                                          self.TSpan.Target, ...
                                          self.TSpan.Regular, ...
                                          self.FDA.FdParamsTarget );

            end

        end


        function dlX = decodeDispatcher( self, dlZ, args )
            % Generate X from Z either using forward or predict
            % Subclasses can override
            arguments
                self            CompactAEModel
                dlZ             dlarray
                args.forward    logical = false
                args.dlX        dlarray % redundant here
            end

            if args.forward
                dlX = forward( self.Nets.Decoder, dlZ );
            else
                dlX = predict( self.Nets.Decoder, dlZ );
            end

        end


        function [ YHat, YHatScore] = predictAuxNet( self, Z )
            % Make prediction from Z using an auxiliary network
            arguments
                self            CompactAEModel
                Z               {mustBeA(Z, {'double', 'dlarray'})}
            end
                
            if isa( Z, 'dlarray' )
                dlZ = Z;
            else
                if size( Z, 1 ) ~= self.ZDim
                    Z = Z';
                end
                dlZ = dlarray( Z, 'CB' );
            end

            auxNet = (self.LossFcnTbl.Types == 'Auxiliary');
            if ~any( auxNet )
                eid = 'aeModel:NoAuxiliaryFunction';
                msg = 'No auxiliary loss function specified in the model.';
                throwAsCaller( MException(eid,msg) );
            end

            if ~self.LossFcnTbl.HasNetwork( auxNet )
                eid = 'aeModel:NoAuxiliaryNetwork';
                msg = 'No auxiliary network specified in the model.';
                throwAsCaller( MException(eid,msg) );
            end

            auxNetName = self.LossFcnTbl.Names( auxNet );

            % get the network's prediction with softmax
            fcLayer = self.Nets.(auxNetName).Layers(end-1).Name;
            outLayer = self.Nets.(auxNetName).Layers(end).Name;

            [ dlYHat, dlYHatScore ] = predict( self.Nets.(auxNetName), ...
                                               dlZ, ...
                                               Outputs = {outLayer, fcLayer} );

            YHat = double(extractdata( dlYHat ));
            YHatScore = double(extractdata( dlYHatScore ));

            YHat = double(onehotdecode( YHat', 1:self.CDim, 2 ))';

        end


        function YHat = predictCompNet( self, thisDataset )
            % Make prediction from X using the comparator network
            arguments
                self            CompactAEModel
                thisDataset     ModelDataset
            end

            dlX = thisDataset.getDLInput( self.XDimLabels );

            if self.FlattenInput && size( dlX, 3 ) > 1
                dlX = flattenDLArray( dlX );
            end

            compNet = (self.LossFcnTbl.Types == 'Comparator');
            if ~any( compNet )
                eid = 'aeModel:NoComparatorFunction';
                msg = 'No comparator loss function specified in the model.';
                throwAsCaller( MException(eid,msg) );
            end

            if ~self.LossFcnTbl.HasNetwork( compNet )
                eid = 'aeModel:NoComparatorNetwork';
                msg = 'No comparator network specified in the model.';
                throwAsCaller( MException(eid,msg) );
            end

            compNetName = self.LossFcnTbl.Names( compNet );

            dlYHat = predict( self.Nets.(compNetName), dlX );

            YHat = double(extractdata( dlYHat ))';
            YHat = double(onehotdecode( YHat, 1:thisDataset.CDim, 2 ));

        end


        function self = incrementActiveZDim( self )
            % Increment the number of active dimensions
            arguments
                self            CompactAEModel
            end

            self.ZDimActive = min( self.ZDimActive + 1, self.ZDim );

        end


        function dlZ = maskZ( self, dlZ )
            % Mask latent codes based on the number of active dimensions
            arguments
                self            CompactAEModel
                dlZ             {mustBeA(dlZ, {'double', 'dlarray'})}
            end

            for i = self.ZDimActive+1:self.ZDim
                dlZ(i,:) = 0;
            end

        end


        function save( self )
            % Save the model plots and the object itself
            arguments
                self            CompactAEModel
            end

            if self.ShowPlots
                plotObjects = self.Axes;
                plotObjects.Components = self.Figs.Components;
                plotObjects.LossFig = self.Trainer.LossFig;
    
                savePlots( plotObjects, self.Info.Path, self.Info.Name );
            end
            
        end


        function self = compress( self, level )
            % Clear the objects to save memory
            % including object specific to an AE
            arguments
                self            CompactAEModel
                level           double ...
                    {mustBeInRange( level, 0, 3 )} = 0
            end

            self = compress@CompactRepresentationModel( self, level );

            if level >= 1
                self.Trainer.LossFig = [];
                self.Trainer.LossLines = [];
            end

            if level == 3
                self.Optimizer = [];
            end

        end

    end


    methods (Static)

        function [ eval, pred, cor ] = evaluateSet( self, thisDataset )
            % Evaluate the model with a specified dataset
            % doing additional work to the superclass method
            arguments
                self             CompactAEModel
                thisDataset      ModelDataset
            end

            % call the superclass method
            [ eval, pred, cor ] = ...
                evaluateSet@CompactRepresentationModel( self, thisDataset );

            if any(self.LossFcnTbl.Types == 'Comparator')
                % compute the comparator loss using the comparator network
                pred.ComparatorYHat = predictCompNet( self, thisDataset );
                eval.Comparator = evaluateClassifier( pred.Y, pred.ComparatorYHat );
            end
    
            if any(self.LossFcnTbl.Types == 'Auxiliary')
                % compute the auxiliary loss using the network
                pred.AuxNetworkYHat = predictAuxNet( self, pred.Z );
                eval.AuxNetwork = evaluateClassifier( pred.Y, pred.AuxNetworkYHat );
            end

            eval = flattenStruct( eval );
        
        end

    end


end




