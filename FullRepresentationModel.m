classdef FullRepresentationModel
    % Super class encompassing all cross-validated dimensional reduction models

    properties
        XDim            % X dimension (number of points)
        ZDim            % Z dimension (number of features)
        CDim            % C dimension (number of classes)
        XChannels       % number of channels in X
        Scale           % scaling factor for reconstruction loss
        AuxModelType    % type of auxiliary model to use
        KFolds          % number of cross validation partitions
        Partitions      % logical array specifying the train/validation split
        SubModels       % array of trained models
        Evaluations     % evaluations structure for the sub-models
        ShowPlots       % flag whether to show plots
        Figs            % figures holding the plots
        Axes            % axes for plotting latent space and components
        NumCompLines    % number of lines in the component plot
    end

    methods

        function self = FullRepresentationModel( args )
            % Initialize the model
            arguments
                args.XDim           double ...
                    {mustBeInteger, mustBePositive} = 10
                args.ZDim           double ...
                    {mustBeInteger, mustBePositive} = 1
                args.CDim           double ...
                    {mustBeInteger, mustBePositive} = 1
                args.XChannels      double ...
                    {mustBeInteger, mustBePositive} = 1
                args.auxModelType   string ...
                        {mustBeMember( args.auxModelType, ...
                        {'Logistic', 'Fisher', 'SVM'} )} = 'Logistic'
                args.KFold      double ...
                    {mustBeInteger, mustBePositive} = 5
                args.NumCompLines   double...
                    {mustBeInteger, mustBePositive} = 8
                args.ShowPlots      logical = true
            end

            self.XDim = args.XDim;
            self.ZDim = args.ZDim;
            self.CDim = args.CDim;
            self.XChannels = args.XChannels;
            self.AuxModelType = args.auxModelType;
            self.KFolds = args.KFold;
            self.NumCompLines = args.NumCompLines;

            self.ShowPlots = args.ShowPlots;

            if args.ShowPlots
                [self.Figs, self.Axes] = ...
                        initializePlots( self.XChannels, self.ZDim );
            end

        end


        function self = train( self, thisDataset )
            % Train the model on the data provided using cross validation
            arguments
                self            FullRepresentationModel
                thisDataset     modelDataset
            end

            % re-partition the data to create training and validation sets
            self.Partitions = thisDataset.getCVPartition( thisDataset, ...
                                        KFold = self.KFolds );
            
            % run the cross validation loop
            for k = 1:self.KFolds
            
                % set the kth partitions
                thisTrnSet = thisDataset.partition( self.Partitions(:,k) );
                thisValSet = thisDataset.partition( ~self.Partitions(:,k) );
                
                % initialize the sub-model
                self.SubModels{k} = self.initSubModel;

                % train the sub-model
                self.SubModels{k} = self.SubModels{k}.train( ...
                                            thisTrnSet, thisValSet );

                % evaluate the sub-model
                self.Evaluations.Training(k) = ...
                                self.SubModels{k}.evaluate( thisTrnSet );
                self.Evaluations.Validation(k) = ...
                                self.SubModels{k}.evaluate( thisValSet );

            end

        end


        function encodeCV( self, X )
            % Encode aggregated features Z from X using all models
            arguments
                self        FullRepresentationModel
                X           double
            end

        end


        function reconstructCV( self, Z )
            % Reconstruct aggregated X from Z using all models
            arguments
                self        FullRepresentationModel
                Z           double
            end


        end

    end



end


function [figs, axes]= initializePlots( XChannels, ZDim )
    % Setup plots for latent space and components
   
    % setup figure for Z distribution and clustering
    figs.LatentSpace = figure(1);
    clf;
    axes.ZDistribution = subplot( 1, 2, 1 );
    axes.ZClustering = subplot( 1, 2, 2 );

    % setup the components figure
    figs.Components = figure(2);
    figs.Components.Position(2) = 0;
    figs.Components.Position(3) = 100 + ZDim*250;
    figs.Components.Position(4) = 50 + XChannels*200;
    
    clf;
    axes.Comp = gobjects( XChannels, ZDim );

    for i = 1:XChannels
        for j = 1:ZDim
            axes.Comp(i,j) = subplot( XChannels, ZDim, (i-1)*ZDim + j );
        end
    end

end
