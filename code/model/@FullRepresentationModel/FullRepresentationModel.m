classdef FullRepresentationModel
    % Super class encompassing all cross-validated dimensional reduction models

    properties
        XInputDim       % X dimension (number of points) for input
        XTargetDim      % X dimension for output
        ZDim            % Z dimension (number of features)
        CDim            % C dimension (number of classes)
        XChannels       % number of channels in X
        TSpan           % time-spans used in fitting
        FDA             % functional data parameters used in fitting
        Info            % information about the dataset
        Scale           % scaling factor for reconstruction loss
        AuxModelType    % type of auxiliary model to use
        KFolds          % number of cross validation partitions
        Partitions      % logical array specifying the train/validation split
        IdenticalPartitions % flag for special case of identical partitions
        SubModels       % array of trained models
        MeanCurve       % estimated mean curve
        Loss            % collated losses from sub-models
        CVLoss          % aggregate cross-validated losses
        CVAuxiliary     % aggregate paramteres for auxiliary models
        CVCorrelation   % mean correlation matrices
        CVLatentComponents % cross-validated latent components
        ComponentOrder  % optimal arrangement of sub-model components
        ComponentDiffRMSE % overall difference between sub-models
        ShowPlots       % flag whether to show plots
        Figs            % figures holding the plots
        Axes            % axes for plotting latent space and components
        NumCompLines    % number of lines in the component plot
        RandomSeed      % for reproducibility
        RandomSeedResets % whether to reset the seed for each sub-model
        CompressionLevel % degree of compression when saving
                         % 0 = None
                         % 1 = Clear Figures
                         % 2 = Clear Predictions
                         % 3 = Clear Optimizer

    end

    methods

        function self = FullRepresentationModel( thisDataset, args )
            % Initialize the model
            arguments
                thisDataset             ModelDataset
                args.ZDim               double ...
                    {mustBeInteger, mustBePositive}
                args.AuxModelType       string ...
                        {mustBeMember( args.AuxModelType, ...
                        {'Logistic', 'Fisher', 'SVM'} )} = 'Logistic'
                args.KFolds             double ...
                    {mustBeInteger, mustBePositive} = 5
                args.RandomSeed         double ...
                    {mustBeInteger, mustBePositive}
                args.RandomSeedResets   logical = false;
                args.NumCompLines       double...
                    {mustBeInteger, mustBePositive} = 5
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
            self = self.setScalingFactor( thisDataset.XTarget );

            self.ZDim = args.ZDim;
            self.AuxModelType = args.AuxModelType;
            self.KFolds = args.KFolds;
            self.IdenticalPartitions = args.IdenticalPartitions;
            self.SubModels = cell( self.KFolds, 1 );

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

            if args.ShowPlots
                [self.Figs, self.Axes] = ...
                        initializePlots( self.XChannels, self.ZDim );
            end

        end

        % class methods

        self = arrangeComponents( self )

        self = computeCVCorrelations( self )

        self = computeCVLosses( self )

        self = computeCVParameters( self )

        self = conserveMemory( self, level )

        [ ZFold, ZMean, ZSD ] = encode( self, thisDataset )

        [ eval, pred, cor ] = evaluateSet( self, thisData )

        self = plotAllALE( self, arg )

        self = plotAllLatentComponents( self, arg )

        [ YHatFold, YHatMaj ] = predictAux( self, Z )

        [ XHatFold, XHatMean, XHatSD ] = reconstruct( self, Z )

        save( self )

        self = setLatentComponents( self )

        self = setScalingFactor( self, data )

        self = train( self, thisDataset )

    end

    methods (Static)

        P = calcCVParameter( subModels, param )

        P = calcCVNestedParameter( subModels, param )


    end


end