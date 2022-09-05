classdef SubRepresentationModel
    % Super class encompassing all individual dimensional reduction models

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
        AuxModel        % auxiliary model
        AuxModelZMean   % mean used in standardizing Z prior to fitting (apply before prediction)
        AuxModelZStd    % standard deviation used prior to fitting (apply before prediction)
        AuxModelALE     % auxiliary model's Accumulated Local Effects
        ALEQuantiles    % quantiles of Z used in computing ALE

        ShowPlots       % flag whether to show plots
        Figs            % figures holding the plots
        Axes            % axes for plotting latent space and components
        NumCompLines    % number of lines in the component plot

        MeanCurve       % estimated mean curve
        ComponentType   % type of components generated (Mean or PDP)
        LatentComponents % computed components across partitions

        Predictions      % training and validation predictions
        Loss             % training and validation losses
        Correlations     % training and validation correlations
    end

    methods

        function self = SubRepresentationModel( theFullModel, fold )
            % Initialize the model
            arguments
                theFullModel        FullRepresentationModel
                fold                double ...
                                    {mustBeInteger, mustBePositive}
            end

            self.XInputDim = theFullModel.XInputDim;
            self.XTargetDim = theFullModel.XTargetDim;
            self.ZDim = theFullModel.ZDim;
            self.CDim = theFullModel.CDim;
            self.XChannels = theFullModel.XChannels;
            self.TSpan = theFullModel.TSpan;
            self.FDA = theFullModel.FDA;
            self.Info = theFullModel.Info;
            self.Scale = theFullModel.Scale;
            self.AuxModelType = theFullModel.AuxModelType;
            self.ShowPlots = theFullModel.ShowPlots;
            self.Figs = theFullModel.Figs;
            self.Axes = theFullModel.Axes;
            self.NumCompLines = theFullModel.NumCompLines;

            self.Info.Name = strcat( self.Info.Name, "-Fold", ...
                                     num2str( fold, "%02d" ) );

        end

        % class methods

        [F, QMid, ZQMid, offsets ] = calcALE( self, dlZ, args )

        [ varProp, compVar ] = calcExplainedVariance( self, X, XC, offsets )

        self = compress( self, level )

        self = evaluate( self, thisTrnSet, thisValSet )

        [auxALE, Q] = getAuxALE( self, thisDataset, args )

        [ XA, Q, XC ] = getLatentResponse( self, thisDataset )

        [ YHat, YHatScore] = predictAuxModel( self, Z )

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