classdef AEModel < RepresentationModel
    % Subclass defining the framework for an autoencoder model
    
    properties
        Nets           % networks defined in this model (structure)
        NetNames       % names of the networks (for convenience)
        NumNetworks    % number of networks
        LossFcns       % array of loss functions
        LossFcnNames   % names of the loss functions
        LossFcnTbl     % convenient table summarising loss function details
        NumLoss        % number of computed losses
        FlattenInput   % whether to flatten input
        HasSeqInput    % supports variable-length input
        IsVAE          % if the model is a variational autoencoder
        NumEncodingDraws % number of draws from the distribution per row
        UseEncodingMean  % whether to use mean output in predictions
        Trainer        % trainer object holding training parameters
        Optimizer      % optimizer object
        InitZDimActive % initial number of Z dimensions active
        ZDimActive     % number of dimensions currently active
        HasCentredDecoder % if the decoder predicts centred X
        UsesDensityEstimation % if the model is based on density estimation
        XComponentDim  % dimensions of the component (may differ from XTargetDim)
        MeanCurveTarget   % mean curve for the X target time span
        AuxNetResponse % auxiliary network effect response
        ActiveCompLossFcn % loss function object to be used in training
    end

    properties (Dependent = true)
        XDimLabels     % dimensional labelling for X input dlarrays
        XNDimLabels    % dimensional labelling for time-normalized output
    end

    methods

        function self = AEModel( thisDataset, ...
                                 superArgs, ...
                                 superArgs2, ...
                                 args )
            % Initialize the model
            arguments
                thisDataset             ModelDataset
                superArgs.?RepresentationModel
                superArgs2.name         string
                superArgs2.path         string
                args.FlattenInput       logical = false
                args.HasSeqInput        logical = false
                args.IsVAE              logical = true
                args.NumEncodingDraws    double ...
                        {mustBeInteger,mustBePositive} = 1
                args.UseEncodingMean     logical = true
                args.InitZDimActive     double ...
                    {mustBeInteger} = 1
                args.HasCentredDecoder  logical = true
                args.LossFcns           struct = []
                args.Trainer            struct = []
                args.Optimizer          struct = []
            end

            tic;

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            superArgs2Cell = namedargs2cell( superArgs2 );
            self = self@RepresentationModel( thisDataset, ...
                                             superArgsCell{:}, ...
                                             superArgs2Cell{:} );
 
            % check dataset is suitable
            if thisDataset.isFixedLength == args.HasSeqInput
                eid = 'AEModel:DatasetNotSuitable';
                if thisDataset.isFixedLength
                    msg = 'The dataset should have variable length for the model.';
                else
                    msg = 'The dataset should have fixed length for the model.';
                end
                throwAsCaller( MException(eid,msg) );
            end

            self.IsVAE = args.IsVAE;
            self.NumEncodingDraws = args.NumEncodingDraws;
            self.UseEncodingMean = args.UseEncodingMean;

            self.FlattenInput = args.FlattenInput;
            self.HasSeqInput = args.HasSeqInput;
            self.HasCentredDecoder = args.HasCentredDecoder;

            if args.InitZDimActive==0
                self.InitZDimActive = self.ZDim;
            else
                self.InitZDimActive = min( args.InitZDimActive, self.ZDim );
            end

            % initialize the loss functions
            self = self.initLossFcns( args.LossFcns );

            % add details associated with the networks
            self.NetNames = {'Encoder', 'Decoder'};
            self.NumNetworks = 2;
            self = self.addLossFcnNetworks;

            % check if active component loss functions are present
            idx = find( self.LossFcnTbl.Types=='Component'  ...
                        & self.LossFcnTbl.DoCalcLoss );
            if ~isempty( idx )
                % identify the active component loss function for training
                % use the first one identified to determine parameters
                self.ActiveCompLossFcn = self.LossFcns.(self.LossFcnTbl.Names(idx(1)));
            end

            % check if trainer arguments include parallel processing
            flds = fields(args.Trainer);
            fldIdx = find(strcmpi(flds, 'inparallel'));
            if ~isempty(fldIdx)
                % set a flag to construct the appropriate trainer object
                useParallelProcessing = args.Trainer.(flds{fldIdx});
                % remove this argument so it is not passed to the constructor
                args.Trainer = rmfield( args.Trainer, flds{fldIdx} );
                if ~useParallelProcessing
                    % remove the doUseGPU field, if present
                    fldIdx = find(strcmpi(flds, 'dousegpu'));
                    if ~isempty(fldIdx)
                        args.Trainer = rmfield( args.Trainer, flds{fldIdx} );
                    end
                end
            else
                useParallelProcessing = false;
            end

            % initialize the trainer
            try
                argsCell = namedargs2cell( args.Trainer );
            catch
                argsCell = {};
            end

            if useParallelProcessing
                self.Trainer = ParallelModelTrainer( self.LossFcnTbl, ...
                                             argsCell{:}, ...
                                             maxBatchSize = thisDataset.NumObs, ...
                                             showPlots = self.ShowPlots );
            else
                self.Trainer = ModelTrainer( self.LossFcnTbl, ...
                                             argsCell{:}, ...
                                             showPlots = self.ShowPlots );
            end

            % initialize the optimizer
            try
                argsCell = namedargs2cell( args.Optimizer );
            catch
                argsCell = {};
            end
            self.Optimizer = ModelOptimizer( self.NetNames, argsCell{:} );

            self.Timing.Training.InitializationTime = toc;

        end


        function labels = get.XDimLabels( self )
            % Get the X dimensional labels for dlarrays
            arguments
                self            AEModel
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
                self            AEModel
            end

            if self.XChannels==1
                labels = 'CB';
            else
                labels = 'SBC';
            end
            
        end

        % class methods

        closeFigures( self )

        [ dlXC, Q, dlZC ] = calcLatentComponents( self, args, args2 )

        dlXCHat = calcALE( self, dlXC, A, w )

        dlXCHat = calcPDP( self, dlXC )

        self = compress( self, level )

        dlZ = encode( self, X, arg )

        [ dlZ, dlXGen, dlXHat, dlXC, state ] = forward( self, encoder, decoder, dlX )
        
        [ dlXHat, state ] = forwardDecoder( self, decoder, dlZ )

        [ dlZ, state ] = forwardEncoder( self, encoder, dlX )

        [ dlX, dlY, dlXN ] = getDLArrays( self, thisDataset )

        self = incrementActiveZDim( self )

        self = initLossFcns( self, setup )

        self = initLossFcnNetworks( self )

        dlZ = maskZ( self, dlZ )

        self = train( self, thisData )

        [ YHat, YHatScore] = predictAuxNet( self, Z )

        YHat = predictCompNet( self, thisDataset )

        [ dlZC, A, w ] = prepALE( self, dlZ, args )

        dlZC = prepPDP( self, dlZ, args )
        
        [ dlXHat, XHatSmth ] = reconstruct( self, Z, args )

        self = setLossInfoTbl( self )

        self = setLossScalingFactor( self )

    end


    methods (Static)

        [ eval, pred, cor ] = evaluateSet( self, thisDataset )

    end

    methods (Abstract)

        net = initEncoder( self )

        net = initDecoder( self )

        setXTargetDim

    end

end