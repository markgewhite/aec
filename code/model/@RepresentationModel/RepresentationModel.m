classdef RepresentationModel
    % Super class encompassing all individual dimensional reduction models

    properties
        XInputDim       % X dimension (number of points) for input
        XTargetDim      % X dimension for output
        ZDim            % Z dimension (number of features)
        ZDimAux         % Z dimension for the auxiliary model
        CDim            % C dimension (number of classes)
        XChannels       % number of channels in X
        TSpan           % time-spans used in fitting
        FDA             % functional data parameters used in fitting
        Info            % information about the dataset
        Scale           % scaling factor for reconstruction loss
        AuxModelType    % type of auxiliary model to use
        AuxModel        % auxiliary model
        AuxModelZMean   % mean used in standardizing Z prior to fitting (apply before prediction)
        AuxModelZStd    % standard deviation used prior to fitting (apply before prediction)
        AuxModelResponse % auxiliary model's effects response (PDP or ALE)
        ResponseQuantiles % Z values used for aux model response

        ShowPlots       % flag whether to show plots
        Figs            % figures holding the plots
        Axes            % axes for plotting latent space and components
        NumCompLines    % number of lines in the component plot

        MeanCurve       % estimated mean curve
        ComponentType   % type of components generated
        ComponentCentering % how to centre the generated components
        LatentComponents % computed components across partitions
        LatentResponseFcn % method to generate output using Z

        Predictions     % training and validation predictions
        Loss            % training and validation losses
        Correlations    % training and validation correlations
        Timing          % training and evaluation execution times

        RandomSeed      % for reproducibility
        RandomSeedResets % whether to reset the seed for each sub-model
        CompressionLevel % degree of compression when saving
                         % 0 = None
                         % 1 = Clear Figures
                         % 2 = Clear Predictions
                         % 3 = Clear Optimizer
    end

    methods

        function self = RepresentationModel( thisDataset, args )
            % Initialize the model
            arguments
                thisDataset             ModelDataset
                args.ZDim               double ...
                    {mustBeInteger, mustBePositive} = 1
                args.ZDimAux            double ...
                    {mustBeInteger, ...
                     mustBeGreaterThanOrEqual(args.ZDimAux,0)} = 0
                args.AuxModelType       string ...
                        {mustBeMember( args.AuxModelType, ...
                        {'Logistic', 'Fisher', 'SVM'} )} = 'Logistic'
                args.KFolds             double ...
                    {mustBeInteger, mustBePositive} = 5
                args.RandomSeed         double ...
                    {mustBeInteger, mustBePositive}
                args.RandomSeedResets   logical = false;
                args.NumCompLines       double...
                    {mustBeInteger, mustBePositive} = 9
                args.ComponentType      string ...
                        {mustBeMember( args.ComponentType, ...
                        {'FPC', 'PDP', 'ALE'} )} = 'PDP'
                args.ComponentCentering string ...
                    {mustBeMember( args.ComponentCentering, ...
                                    {'Z', 'X', 'None'} )} = 'Z'
                args.ShowPlots          logical = true
                args.IdenticalPartitions logical = false
                args.Name               string = "[ModelName]"
                args.Path               string = ""
                args.CompressionLevel   double ...
                    {mustBeInRange(args.CompressionLevel, 0, 3)} = 2
            end

            % set properties based on the data
            self.XInputDim = thisDataset.XInputDim;
            self.XTargetDim = thisDataset.XTargetDim;
            self.CDim = thisDataset.CDim;
            self.XChannels = thisDataset.XChannels;
            self.TSpan = thisDataset.TSpan;
            self.FDA = thisDataset.FDA;
            self.Info = thisDataset.Info;

            % set the scaling factor(s) based on all X
            self.Scale = scalingFactor( thisDataset.XTarget );

            self.ZDim = args.ZDim;

            if args.ZDimAux==0
                self.ZDimAux = self.ZDim;
            else
                if args.ZDimAux <= self.ZDim
                    self.ZDimAux = args.ZDimAux;
                else
                    eid = 'RepModel:ZDimAuxOutsideRange';
                    msg = 'Z dimension for the auxiliary model is outside correct range.';
                throwAsCaller( MException(eid,msg) );
                end
            end

            if self.CDim < 3
                self.AuxModelType = args.AuxModelType;
            else
                self.AuxModelType = 'SVM';
            end

            self.ComponentType = args.ComponentType;
            self.ComponentCentering = args.ComponentCentering;
            
            if isfield( args, 'randomSeed' )
                self.RandomSeed = args.RandomSeed;
            else
                self.RandomSeed = [];
            end
            self.RandomSeedResets = args.RandomSeedResets;

            self.Info.Name = args.Name;
            self.Info.Path = args.Path;

            self.NumCompLines = args.NumCompLines;
            self.CompressionLevel = args.CompressionLevel;

            self.ShowPlots = args.ShowPlots;

            [self.Figs, self.Axes] = initializePlots( self.XChannels, ... 
                                                      self.ZDimAux, ...
                                                      self.ShowPlots );

        end

        % class methods

        [ F, zsMid, ZQMid ] = calcALE( self, dlZ, args )

        [ F, prc, ZQ ] = calcPDP( self, dlZ, args )

        [ varProp, compVar ] = calcExplainedVariance( self, X, XC, offsets )

        [ XC, XMean, offsets ] = calcLatentComponents( self, dlZ, args )

        [F, Q, Z] = calcResponse( self, dlZ, args )

        self = compress( self, level )

        self = evaluate( self, thisTrnSet, thisValSet )

        self = getAuxResponse( self, thisDataset, args )

        XC = getLatentResponse( self, thisDataset )

        plotAuxResponse( self, args )

        plotLatentComp( self, args )

        plotZClusters( self, Z, args )

        plotZDist( self, Z, args )
        
        [ YHat, YHatScore] = predictAuxModel( self, Z )

        showAllPlots( self )

    end

    
    methods (Static)

        [eval, pred, cor] = evaluateSet( thisModel, thisDataset )

    end

    
    methods (Abstract)

        % Train the model on the data provided
        self = train( self, thisDataset )

        % Encode features Z from X using the model - placeholder
        Z = encode( self, X )

        % Reconstruct X from Z using the model - placeholder
        XHat = reconstruct( self, Z )

    end

end