classdef RepresentationModel
    % Super class encompassing all individual dimensional reduction models

    properties
        XInputDim       % X dimension (number of points) for input
        XTargetDim      % X dimension for output
        ZDim            % Z dimension (number of features)
        ZDimAux         % Z dimension for the auxiliary model
        CDim            % C dimension (number of classes)
        CLabels         % unique class labels
        XChannels       % number of channels in X
        TSpan           % time-spans used in fitting
        FDA             % functional data parameters used in fitting
        Info            % information about the dataset
        Scale           % scaling factor for reconstruction loss
        AuxObjective    % whether the objective is classification or regression 
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
        DatasetPlotFcn  % function handle to ModelDataset's plot function

        MeanCurve       % estimated mean curve
        LatentComponents % computed components across partitions
        ShowComponentPts % show the predicted points in plots

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
                args.AuxObjective       string ...
                        {mustBeMember( args.AuxObjective, ...
                        {'Classification', 'Regression'} )} = 'Classification'
                args.AuxModelType       string ...
                        {mustBeMember( args.AuxModelType, ...
                        {'Logistic', 'Fisher', 'SVM', 'LR'} )} = 'Logistic'
                args.KFolds             double ...
                    {mustBeInteger, mustBePositive} = 5
                args.RandomSeed         double ...
                    {mustBeInteger, mustBePositive}
                args.RandomSeedResets   logical = false;
                args.NumCompLines       double...
                    {mustBeInteger, mustBePositive} = 9
                args.ShowPlots          logical = true
                args.ShowComponentPts   logical = false
                args.IdenticalPartitions logical = false
                args.Name               string = "[ModelName]"
                args.Path               string = ""
                args.CompressionLevel   double ...
                    {mustBeInRange(args.CompressionLevel, 0, 3)} = 2
            end

            % set properties based on the data
            self.XInputDim = thisDataset.XDim;
            self.CDim = thisDataset.CDim;
            self.CLabels = single(unique(thisDataset.Y));
            self.XChannels = thisDataset.XChannels;
            self.Info = thisDataset.Info;

            % initialize the time spans
            self.TSpan = thisDataset.TSpan;

            % set the FDA parameters for input
            % (FDA parameters for target and component are set later)
            self.FDA = thisDataset.FDA;

            % store the function handle of the data set's plot function
            self.DatasetPlotFcn = @thisDataset.plot;

            % set the scaling factor(s) based on all X
            self.Scale = scalingFactor( thisDataset.XInput );

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

            self.AuxObjective = args.AuxObjective;

            if self.CDim < 3
                self.AuxModelType = args.AuxModelType;
            else
                self.AuxModelType = 'SVM';
            end
           
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
            self.ShowComponentPts = args.ShowComponentPts;

            [self.Figs, self.Axes] = initializePlots( self.XChannels, ... 
                                                      self.ZDimAux, ...
                                                      self.ShowPlots );

        end

        % class methods

        [ varProp, compVar ] = calcExplainedVariance( self, X, XC, offsets )

        closeFigures( self )

        self = compress( self, level )

        self = evaluate( self, thisTrnSet, thisValSet )

        self = finalizeInit( self, thisDataset )

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

        % Generate the latent components
        [ dlXC, Q, dlZC ] = calcLatentComponents( self, args )


    end

end