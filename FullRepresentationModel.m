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
        CVLatentComponents % cross-validated latent components
        Loss            % collated losses from sub-models
        CVLoss          % aggregate cross-validated losses
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
                self.SubModels{k} = self.SubModels{k}.evaluate( ...
                                            thisTrnSet, thisValSet );
 

            end

            % calculate the aggregated latent components
            self = self.getLatentComponentsCV;

            % calculate the aggregate evaluation across all partitions
            self = self.evaluateCV;    

        end


        function self = evaluateCV( self )
            % Calculate the cross-validated losses
            % using predictions from the sub-models
            arguments
                self        FullRepresentationModel
            end

            self.Loss.Training = collateLosses( self.SubModels, 'Training' );
            self.Loss.Validation = collateLosses( self.SubModels, 'Validation' );

            self.CVLoss.Training = calcCVLoss( self.SubModels, 'Training' );
            self.CVLoss.Validation = calcCVLoss( self.SubModels, 'Validation' );

        end


        function self = getLatentComponentsCV( self )
            % Calculate the cross-validated latent components
            % by averaging across the sub-models
            arguments
                self        FullRepresentationModel
            end

            XC = self.SubModels{1}.LatentComponents;
            for k = 2:self.KFolds
                XC = XC + self.SubModels{k}.LatentComponents;
            end

            self.CVLatentComponents = XC/self.KFolds;


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


        function [ ZFold, ZMean, ZSD ] = encode( self, thisDataset )
            % Encode aggregated features Z from X using all models
            arguments
                self            FullRepresentationModel
                thisDataset     modelDataset
            end

            ZFold = zeros( thisDataset.NumObs, self.ZDim, self.KFolds );
            for k = 1:self.KFolds
                ZFold( :, :, k ) = encode( self.SubModels{k}, thisDataset );
            end
            ZMean = mean( ZFold, 3 );
            ZSD = std( ZFold, [], 3 );

        end


        function [ XHatFold, XHatMean, XHatSD ] = reconstruct( self, Z )
            % Reconstruct aggregated X from Z using all models
            arguments
                self            FullRepresentationModel
                Z               double
            end

            isEnsemble = (size( Z, 3 ) > 1);
            XHatFold = zeros( length(self.TSpan), size(Z,1), self.KFolds );
            for k = 1:self.KFolds
                if isEnsemble
                    XHatFold( :, :, k ) = ...
                            reconstruct( self.SubModels{k}, Z(:,:,k) );
                else
                    XHatFold( :, :, k ) = ...
                            reconstruct( self.SubModels{k}, Z );
                end
            end
            XHatMean = mean( XHatFold, 3 );
            XHatSD = std( XHatFold, [], 3 );
        end


        function [ YHatFold, YHatMaj ] = predictAux( self, Z )
            % Predict Y from Z using all auxiliary models
            arguments
                self            FullRepresentationModel
                Z               double
            end

            isEnsemble = (size( Z, 3 ) > 1);
            nRows = size( Z, 1 );
            YHatFold = zeros( nRows, self.KFolds );
            for k = 1:self.KFolds
                if isEnsemble
                    YHatFold( :, k ) = ...
                            predict( self.SubModels{k}.AuxModel, Z(:,:,k) );
                else
                    YHatFold( :, k ) = ...
                            predict( self.SubModels{k}.AuxModel, Z );
                end
            end

            YHatMaj = zeros( nRows, 1 );
            for i = 1:nRows
                [votes, grps] = groupcounts( YHatFold(i,:)' );
                [ ~, idx ] = max( votes );
                YHatMaj(i) = grps( idx );
            end

        end

    end


end


function aggrLoss = collateLosses( subModels, set )
    % Collate the losses from the submodels
    arguments
        subModels       cell
        set             char ...
            {mustBeMember( set, {'Training', 'Validation'} )}
    end

    nModels = length( subModels );
    thisAggrLoss = zeros( nModels, 1 );
    fields = fieldnames( subModels{1}.Loss.(set) );
    nFields = length( fields );

    for i = 1:nFields
        if length( subModels{1}.Loss.(set).(fields{i}) ) == 1
            for k = 1:nModels
               thisAggrLoss(k) = subModels{k}.Loss.(set).(fields{i});
            end
            aggrLoss.(fields{i}) = thisAggrLoss;
        end
    end

end


function cvLoss = calcCVLoss( subModels, set )
    % Calculate the aggregate cross-validated losses across all submodels
    % drawing on the pre-computed predictions 
    arguments
        subModels       cell
        set             char ...
            {mustBeMember( set, {'Training', 'Validation'} )}
    end

    nModels = length( subModels );

    pairs = [   {'XTarget', 'XHat'}; ...
                {'XTarget', 'XHatSmoothed'}; ...
                {'XRegular', 'XHatRegular'}; ...
                {'Y', 'AuxModelYHat'} ];
    fieldsToPermute = { 'XTarget', 'XHat', 'XHatSmoothed', 'XRegular', 'XHatRegular' };
    fieldsForAuxLoss = { 'AuxModelYHat' };

    nPairs = length( pairs );
    fields = unique( pairs );

    % aggregate all the predictions for each field into one array
    for i = 1:length(fields)

        aggr.(fields{i}) = [];
        doPermute = ismember( fields{i}, fieldsToPermute );

        for k = 1:nModels

            data = subModels{k}.Predictions.(set).(fields{i});
            if doPermute
                data = permute( data, [2 1 3] );
            end
            
            aggr.(fields{i}) = [ aggr.(fields{i}); data ];

        end
    end


    for i = 1:nPairs

        if ismember( pairs{i,2}, fieldsForAuxLoss )
            % cross entropy loss
            cvLoss.(pairs{i,2}) = getPropCorrect( ...
                                          aggr.(pairs{i,1}), ...
                                          aggr.(pairs{i,2}) );
        else
            % mean squared error loss
            cvLoss.(pairs{i,2}) = getReconLoss( subModels{1}, ...
                                          aggr.(pairs{i,1}), ...
                                          aggr.(pairs{i,2}) );
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
