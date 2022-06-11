classdef FullRepresentationModel
    % Super class encompassing all cross-validated dimensional reduction models

    properties
        XInputDim       % X dimension (number of points) for input
        XTargetDim      % X dimension for output
        ZDim            % Z dimension (number of features)
        CDim            % C dimension (number of classes)
        XChannels       % number of channels in X
        Scale           % scaling factor for reconstruction loss
        AuxModelType    % type of auxiliary model to use
        KFolds          % number of cross validation partitions
        Partitions      % logical array specifying the train/validation split
        SubModels       % array of trained models
        Evaluations     % evaluations structure for the sub-models
        EvaluationsCV   % aggregate cross-validated evaluations
        ShowPlots       % flag whether to show plots
        Figs            % figures holding the plots
        Axes            % axes for plotting latent space and components
        NumCompLines    % number of lines in the component plot
    end

    methods

        function self = FullRepresentationModel( thisDataset, args )
            % Initialize the model
            arguments
                thisDataset         modelDataset
                args.ZDim           double ...
                    {mustBeInteger, mustBePositive}
                args.auxModelType   string ...
                        {mustBeMember( args.auxModelType, ...
                        {'Logistic', 'Fisher', 'SVM'} )} = 'Logistic'
                args.KFold      double ...
                    {mustBeInteger, mustBePositive} = 5
                args.NumCompLines   double...
                    {mustBeInteger, mustBePositive} = 8
                args.ShowPlots      logical = true
            end

            self.XInputDim = thisDataset.XInputDim;
            self.XTargetDim = thisDataset.XTargetDim;
            self.CDim = thisDataset.CDim;
            self.XChannels = thisDataset.XChannels;

            self.ZDim = args.ZDim;
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
                    modelEvaluation.evaluate( self.SubModels{k}, thisTrnSet );  
                self.Evaluations.Validation(k) = ...
                    modelEvaluation.evaluate( self.SubModels{k}, thisValSet );  

            end

            % calculate the aggregate evaluation across all partitions
            self.Evaluations.Training = struct2table( self.Evaluations.Training );
            self.Evaluations.Validation = struct2table( self.Evaluations.Validation );

            self.EvaluationsCV.Training = ...
                                evaluate( self.Evaluations.Training );
            self.EvaluationsCV.Validation = ...
                                evaluate( self.Evaluations.Validation );
            

        end


        function loss = getReconLoss( self, X, XHat )
            % Calculate the reconstruction loss
            arguments
                self        FullRepresentationModel
                X           double
                XHat        double
            end

            loss = getReconLoss( self.SubModels{1}, X, XHat );
    
        end


        function loss = getReconTemporalLoss( self, X, XHat )
            % Calculate the reconstruction loss across time domain
            arguments
                self        FullRepresentationModel
                X           double
                XHat        double
            end

            loss = getReconTemporalLoss( self.SubModels{1}, X, XHat );
    
        end


        function [ Z, ZSD ] = encode( self, thisDataset )
            % Encode aggregated features Z from X using all models
            arguments
                self            FullRepresentationModel
                thisDataset     modelDataset
            end

            ZFold = zeros( thisDataset.NumObs, self.ZDim, self.KFolds );
            for k = 1:self.KFolds
                ZFold( :, :, k ) = encode( self.SubModels{k}, thisDataset );
            end
            Z = mean( ZFold, 3 );
            ZSD = std( ZFold, [], 3 );

        end


        function [ XHat, XHatSD ] = reconstruct( self, Z )
            % Reconstruct aggregated X from Z using all models
            arguments
                self            FullRepresentationModel
                Z               double
            end

            XHatFold = zeros( length(self.TSpan), size(Z,1), self.KFolds );
            for k = 1:self.KFolds
                XHatFold( :, :, k ) = reconstruct( self.SubModels{k}, Z );
            end
            XHat = mean( XHatFold, 3 );
            XHatSD = std( XHatFold, [], 3 );
        end

    end 


end


function evalsCV = evaluate( evals )
    % Calculate the aggregate cross-validated evaluations
    % averaging the sub-model evaluations
    arguments
        evals       table
    end

    fields = evals.Properties.VariableNames;

    for i = 1:length( fields )

        Q = evals.(fields{i});
        
        switch fields{i}
            case {'AuxModelYHat', 'AuxNetworkYHat'}
                % Majority vote
                
            
            otherwise
                % Average
                switch fields{i}
                    case {'Z', 'VarProp', 'CompVar', ...
                          'ReconLoss', 'ReconLossSmoothed', ...
                          'ReconLossRegular', ...
                          'AuxModelLoss' }
                        d = 1;
                    case {'XC'}
                        d = 3;
                    otherwise
                        d = 2;
                end
        
                if iscell( Q )
                    Q = cat( d, Q{:} );
                end
        
                evalsCV.mean.(fields{i}) = mean( Q, d );
                evalsCV.sd.(fields{i}) = std( Q, [], d );

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
