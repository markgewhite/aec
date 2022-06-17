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
        ComponentType   % type of components generated (Mean or PDP)
        LatentComponents % cross-validated latent components
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
                thisDataset         ModelDataset
                args.ZDim           double ...
                    {mustBeInteger, mustBePositive}
                args.auxModelType   string ...
                        {mustBeMember( args.auxModelType, ...
                        {'Logistic', 'Fisher', 'SVM'} )} = 'Logistic'
                args.KFolds         double ...
                    {mustBeInteger, mustBePositive} = 5
                args.componentType  char ...
                    {mustBeMember(args.componentType, ...
                        {'Mean', 'PDP'} )} = 'PDP'
                args.NumCompLines   double...
                    {mustBeInteger, mustBePositive} = 8
                args.ShowPlots      logical = true
                args.IdenticalPartitions logical = false
                args.name           string = "[ModelName]"
                args.path           string = ""
            end

            % set properties based on the data
            self.XInputDim = thisDataset.XInputDim;
            self.XTargetDim = thisDataset.XTargetDim;
            self.CDim = thisDataset.CDim;
            self.XChannels = thisDataset.XChannels;
            self.TSpan = thisDataset.TSpan;
            self.FDA = thisDataset.FDA;
            self.Info = thisDataset.Info;
            self.Info.Name = args.name;
            self.Info.Path = args.path;
            
            % set the scaling factor(s) based on all X
            self = self.setScalingFactor( thisDataset.XTarget );

            self.ZDim = args.ZDim;
            self.AuxModelType = args.auxModelType;
            self.KFolds = args.KFolds;
            self.IdenticalPartitions = args.IdenticalPartitions;
            self.SubModels = cell( self.KFolds, 1 );

            self.ComponentType = args.componentType;
            self.NumCompLines = args.NumCompLines;

            self.ShowPlots = args.ShowPlots;

            if args.ShowPlots
                [self.Figs, self.Axes] = ...
                        initializePlots( self.XChannels, self.ZDim );
            end

        end


        function self = setScalingFactor( self, data )
            % Set the scaling factors for reconstructions
            arguments
                self            FullRepresentationModel
                data            double
            end
            
            % set the channel-wise scaling factor
            self.Scale = squeeze(mean(var( data )))';

        end


        function self = train( self, thisDataset )
            % Train the model on the data provided using cross validation
            arguments
                self            FullRepresentationModel
                thisDataset     ModelDataset
            end

            % re-partition the data to create training and validation sets
            self.Partitions = thisDataset.getCVPartition( ...
                                        KFold = self.KFolds, ...
                                        Identical = self.IdenticalPartitions );
            
            % run the cross validation loop
            for k = 1:self.KFolds
            
                % set the kth partitions
                thisTrnSet = thisDataset.partition( self.Partitions(:,k) );
                thisValSet = thisDataset.partition( ~self.Partitions(:,k) );
                
                % initialize the sub-model
                self.SubModels{k} = self.initSubModel( k );

                % train the sub-model
                self.SubModels{k} = self.SubModels{k}.train( thisTrnSet );

                % evaluate the sub-model
                self.SubModels{k} = self.SubModels{k}.evaluate( ...
                                            thisTrnSet, thisValSet );

                % save the model
                self.SubModels{k}.save;

                % clear graphics objects to save memory
                self.SubModels{k} = self.SubModels{k}.clearGraphics;
 
            end

            % calculate the aggregated latent components
            self = self.setLatentComponents;

            % calculate the cross-validated losses
            self = self.computeLosses;

            % save the full model
            self.save;

        end


        function self = computeLosses( self )
            % Calculate the cross-validated losses
            % using predictions from the sub-models
            arguments
                self        FullRepresentationModel
            end

            self.Loss.Training = collateLosses( self.SubModels, 'Training' );
            self.CVLoss.Training = calcCVLoss( self.SubModels, 'Training' );

            if isfield( self.SubModels{1}.Loss, 'Validation' )
                self.Loss.Validation = collateLosses( self.SubModels, 'Validation' );
                self.CVLoss.Validation = calcCVLoss( self.SubModels, 'Validation' );
            end

        end


        function self = setLatentComponents( self )
            % Calculate the cross-validated latent components
            % by averaging across the sub-models
            arguments
                self        FullRepresentationModel
            end

            XC = self.SubModels{1}.LatentComponents;
            for k = 2:self.KFolds
                XC = XC + self.SubModels{k}.LatentComponents;
            end

            self.LatentComponents = XC/self.KFolds;


        end


        function save( self )
            % Save the model plots and the object itself
            arguments
                self            FullRepresentationModel
            end

            if self.ShowPlots

                plotObjects = self.Axes;
                plotObjects.Components = self.Figs.Components;   
                savePlots( plotObjects, self.Info.Path, self.Info.Name );

            end

            model = self.clearGraphics;
            filename = strcat( self.Info.Name, "-FullModel" );
            save( fullfile( self.Info.Path, filename ), 'model' );

        end


        function obj = clearGraphics( obj )
            % Clear the graphics objects to save memory
            arguments
                obj            FullRepresentationModel
            end

            obj.Figs = [];
            obj.Axes = [];

        end


        function [ ZFold, ZMean, ZSD ] = encode( self, thisDataset )
            % Encode aggregated features Z from X using all models
            arguments
                self            FullRepresentationModel
                thisDataset     ModelDataset
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
            XHatFold = zeros( self.XTargetDim, size(Z,1), self.KFolds );
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


        function [ eval, pred ] = evaluateSet( self, thisData )
            % Evaluate the full model using the routines in compact model
            arguments
                self            FullRepresentationModel
                thisData        ModelDataset
            end

            [ eval, pred ] = self.SubModels{1}.evaluateSet( ...
                                    self.SubModels{1}, thisData );

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
    fields = fieldnames( subModels{1}.Loss.(set) );
    nFields = length( fields );

    for i = 1:nFields

        fldDim = size( subModels{1}.Loss.(set).(fields{i}) );
        thisAggrLoss = zeros( [nModels fldDim] );
        for k = 1:nModels
           thisAggrLoss(k,:,:) = subModels{k}.Loss.(set).(fields{i});
        end
        aggrLoss.(fields{i}) = thisAggrLoss;

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

    scale = mean(cellfun( @(m) m.Scale, subModels ));

    for i = 1:nPairs

        A = aggr.(pairs{i,1});
        AHat = aggr.(pairs{i,2});

        if ismember( pairs{i,2}, fieldsForAuxLoss )
            % cross entropy loss
            cvLoss.(pairs{i,2}) = getPropCorrect( A, AHat );
        else
            % mean squared error loss
            cvLoss.(pairs{i,2}) = reconLoss( A, AHat, scale );
            % temporal mean squared error loss
            cvLoss.([pairs{i,2} 'TemporalMSE']) = ...
                                    reconTemporalLoss( A', AHat', scale );

            % temporal bias
            cvLoss.([pairs{i,2} 'TemporalBias']) = ...
                                    reconTemporalBias( A', AHat', scale );

            % temporal bias
            A0 = AHat' - cvLoss.([pairs{i,2} 'TemporalBias']);
            cvLoss.([pairs{i,2} 'TemporalVar']) = ...
                                    reconTemporalLoss( A', A0, scale );
        
        end
    
    end



end

