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

            if ~isempty( self.RandomSeed )
                % set random seed for reproducibility
                rng( self.RandomSeed );
            end

            % re-partition the data to create training and validation sets
            self.Partitions = thisDataset.getCVPartition( ...
                                        KFold = self.KFolds, ...
                                        Identical = self.IdenticalPartitions );
            
            % run the cross validation loop
            for k = 1:self.KFolds
            
                if self.ShowPlots
                    disp(['Fold ' num2str(k) '/' num2str(self.KFolds)]);
                end
                
                % set the kth partitions
                thisTrnSet = thisDataset.partition( self.Partitions(:,k) );
                thisValSet = thisDataset.partition( ~self.Partitions(:,k) );
                
                % initialize the sub-model
                self = self.initSubModel( k );

                if self.RandomSeedResets && ~isempty( self.RandomSeed )
                    % reset the random seed for the submodel
                    rng( self.RandomSeed );
                end

                % train the sub-model
                self.SubModels{k} = self.SubModels{k}.train( thisTrnSet );

                % evaluate the sub-model
                self.SubModels{k} = self.SubModels{k}.evaluate( ...
                                            thisTrnSet, thisValSet );

                % save the model
                self.SubModels{k}.save;
 
            end

            % set the overall mean curve
            self.MeanCurve = thisDataset.XInputRegularMean;

            % find the optimal arrangement of sub-model components
            if self.KFolds > 1
                % NEED TO FIX
                % self = self.arrangeComponents;
            end

            % calculate the aggregated latent components
            self = self.setLatentComponents;

            % calculate the cross-validated losses
            self = self.computeCVLosses;

            % calculate cross-validated correlation matrices
            self = self.computeCVCorrelations;

            % calculate other cross-validated parameters
            self = self.computeCVParameters;

            if self.ShowPlots
                self.plotAllLatentComponents;
                self.plotAllALE;
            end
            
            % save the full model
            self.save;

        end


        function self = computeCVLosses( self )
            % Calculate the cross-validated losses
            % using predictions from the sub-models
            arguments
                self        FullRepresentationModel
            end

            self.Loss.Training = self.collateLosses( self.SubModels, 'Training' );
            self.CVLoss.Training = self.calcCVLoss( self.SubModels, 'Training' );

            if isfield( self.SubModels{1}.Loss, 'Validation' )
                self.Loss.Validation = self.collateLosses( self.SubModels, 'Validation' );
                self.CVLoss.Validation = self.calcCVLoss( self.SubModels, 'Validation' );
            end

        end


        function self = computeCVCorrelations( self )
            % Average the correlation matrices across sub-models
            arguments
                self        FullRepresentationModel
            end

            self.CVCorrelation.Training = ...
                self.calcCVCorrelations( self.SubModels, 'Training' );

            if isfield( self.SubModels{1}.Loss, 'Validation' )
                self.CVCorrelation.Validation = ...
                    self.calcCVCorrelations( self.SubModels, 'Validation' );
            end

        end


        function self = computeCVParameters( self )
            % Average specific parameters across sub-models
            arguments
                self        FullRepresentationModel
            end

            self.CVAuxiliary.AuxModelBeta = self.calcCVNestedParameter( ...
                        self.SubModels, {'AuxModel', 'Beta'} );
            self.CVAuxiliary.AuxModelALE = self.calcCVParameter( ...
                        self.SubModels, 'AuxModelALE' );

        end


        function self = arrangeComponents( self )
            % Find the optimal arrangement for the sub-model's components
            % by finding the best set of permutations
            arguments
                          self        FullRepresentationModel
            end

            permOrderIdx = perms( 1:self.ZDim );
            lb = [ length(permOrderIdx) ones( 1, self.KFolds-1 ) ];
            ub = length(permOrderIdx)*ones( 1, self.KFolds );
            options = optimoptions( 'ga', ...
                                    'PopulationSize', 400, ...
                                    'EliteCount', 80, ...
                                    'MaxGenerations', 300, ...
                                    'MaxStallGenerations', 150, ...
                                    'FunctionTolerance', 1E-6, ...
                                    'UseVectorized', true, ...
                                    'PlotFcn', {'gaplotbestf','gaplotdistance', ...
                                                'gaplotbestindiv' } );

            % pre-compile latent components across the sub-models for speed
            latentComp = zeros( self.XInputDim, self.NumCompLines, ...
                                self.ZDim, self.XChannels, self.KFolds );
            for k = 1:self.KFolds
                latentComp(:,:,:,:,k) = self.SubModels{k}.LatentComponents;
            end
            
            % setup the objective function
            objFcn = @(p) arrangementError( p, latentComp, self.ZDim );
            
            % run the genetic algorithm optimization
            [ componentPerms, componentMSE ] = ...
                                ga( objFcn, self.KFolds, [], [], [], [], ...
                                    lb, ub, [], 1:self.KFolds, options );
        
            % generate the order from list of permutations
            self.ComponentOrder = zeros( self.KFolds, self.ZDim );
            for k = 1:self.KFolds
                self.ComponentOrder( k, : ) = permOrderIdx( componentPerms(k), : );
            end
            self.ComponentDiffRMSE = sqrt( componentMSE );

        end


        function self = setLatentComponents( self )
            % Calculate the cross-validated latent components
            % by averaging across the sub-models
            arguments
                self        FullRepresentationModel
            end

            XC = self.SubModels{1}.LatentComponents;
            for k = 2:self.KFolds

                if isempty( self.ComponentOrder )
                    % use model arrangement
                    comp = self.SubModels{k}.LatentComponents;
                else
                    % use optimized arrangement
                    comp = reshape( self.SubModels{k}.LatentComponents, ...
                                self.XInputDim, self.NumCompLines, [] );
                    comp = comp( :, :, self.ComponentOrder(k,:) );
                    comp = reshape( comp, self.XInputDim, [] );
                end

                XC = XC + comp;
            
            end

            self.CVLatentComponents = XC/self.KFolds;

        end


        function self = plotAllLatentComponents( self, arg )
            % Plot all the latent components from the sub-models
            arguments
                self                FullRepresentationModel
                arg.Rearranged      logical = false
            end

            figs = gobjects( self.KFolds, 1 );
            axes = gobjects( self.XChannels, self.ZDim, self.KFolds );
            for c = 1:self.XChannels

                % create a temporary large figure
                figs(c) = figure;
                figs(c).Visible = false;
                figs(c).Position(3) = 100 + self.ZDim*250;
                figs(c).Position(4) = 50 + self.KFolds*200;

                % create all subplots
                for k = 1:self.KFolds
                    for d = 1:self.ZDim
                        axes( c, d, k ) = ...
                            subplot( self.KFolds, self.ZDim, ...
                                 (k-1)*self.ZDim + d );
                    end
                end

            end

            % plot all the components across figures
            for k = 1:self.KFolds
                if arg.Rearranged
                    plotLatentComp( self.SubModels{k}, ...
                                order = self.ComponentOrder(k,:), ...
                                axes = axes(:,:,k), ...
                                showTitle = (k==1), ...
                                showLegend = false, ...
                                showXAxis = (k==self.KFolds) );
                else
                    plotLatentComp( self.SubModels{k}, ...
                                axes = axes(:,:,k), ...
                                showTitle = (k==1), ...
                                showLegend = false, ...
                                showXAxis = (k==self.KFolds) );
                end
            end

            % save the figures and then close
            name = strcat( self.Info.Name, 'AllKFolds' );
            for c = 1:self.XChannels
                figComp.Components = figs(c);
                savePlots( figComp, self.Info.Path, name );
                close( figs(c) );
            end

        end   


        function self = plotAllALE( self, arg )
            % Plot all the ALE curves from the sub-models
            arguments
                self            FullRepresentationModel
                arg.type        char {mustBeMember( ...
                    arg.type, {'Model', 'Network'} )} = 'Model'
            end

            % create a temporary figure
            fig = figure;
            fig.Visible = false;
            fig.Position(3) = 50 + self.KFolds*200;

            % create all subplots
            axes = gobjects( self.KFolds, 1 );
            for k = 1:self.KFolds
                axes( k ) = subplot( 1, self.KFolds, k );
            end

            % plot all the components across figures
            for k = 1:self.KFolds
                plotALE( self.SubModels{k}, type = arg.type, ...
                         axis = axes(k) );
            end

            % save the figures and then close
            name = strcat( self.Info.Name, 'AllModelALE-', arg.type );
            figALE.ALE = fig;
            savePlots( figALE, self.Info.Path, name );
            close( fig );

        end


        function save( self )
            % Save the model plots and the object itself
            arguments
                self            FullRepresentationModel
            end

            filename = strcat( self.Info.Name, "-FullModel" );
            
            theModel = self;
            if self.CompressionLevel >= 1
                theModel.Figs = [];
                theModel.Axes = [];
            end
            for k = 1:theModel.KFolds
                theModel.SubModels{k} = ...
                    theModel.SubModels{k}.compress( self.CompressionLevel );
            end

            save( fullfile( self.Info.Path, filename ), 'theModel' );

        end


        function self = conserveMemory( self, level )
            % Conserve memory usage
            arguments
                self            FullRepresentationModel
                level           double ...
                    {mustBeInRange( level, 0, 3 )} = 0
            end

            if level >= 1
                self.Figs = [];
                self.Axes = [];
            end

            for k = 1:self.KFolds
                self.SubModels{k} = ...
                    self.SubModels{k}.compress( self.CompressionLevel );
            end

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


        function [ eval, pred, cor ] = evaluateSet( self, thisData )
            % Evaluate the full model using the routines in compact model
            arguments
                self            FullRepresentationModel
                thisData        ModelDataset
            end

            [ eval, pred,cor ] = self.SubModels{1}.evaluateSet( ...
                                    self.SubModels{1}, thisData );

        end


    end


    methods (Static)

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
                        {'Y', 'AuxModelYHat'}; ];
            fieldsToPermute = { 'XTarget', 'XHat', 'XHatSmoothed', 'XRegular', 'XHatRegular' };
            fieldsForAuxLoss = { 'AuxModelYHat', 'AuxNetworkYHat', 'ComparatorYHat' };
        
            % check if fields are present
            fields = unique( pairs );
            nPairs = length( pairs );
            for i = 1:nPairs
                if ~isfield( subModels{1}.Predictions.(set), fields{i} )
                    pairs{i,:} = [];
                end
            end
            fields = unique( pairs );
            nPairs = length( pairs );
        
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
        
            scale = mean( cell2mat(cellfun( ...
                            @(m) m.Scale, subModels, UniformOutput = false )), 1 );
        
            for i = 1:nPairs
        
                A = aggr.(pairs{i,1});
                AHat = aggr.(pairs{i,2});
        
                if ismember( pairs{i,2}, fieldsForAuxLoss )
                    % cross entropy loss
                    cvLoss.(pairs{i,2}) = evaluateClassifier( A, AHat );
        
                else
                    % mean squared error loss
                    cvLoss.(pairs{i,2}) = reconLoss( A, AHat, scale );
        
                    % permute dimensions for temporal losses
                    A = permute( A, [2 1 3] );
                    AHat = permute( AHat, [2 1 3] );
        
                    % temporal mean squared error loss
                    cvLoss.([pairs{i,2} 'TemporalMSE']) = ...
                                            reconTemporalLoss( A, AHat, scale );
        
                    % temporal bias
                    cvLoss.([pairs{i,2} 'TemporalBias']) = ...
                                            reconTemporalBias( A, AHat, scale );
        
                    % temporal variance rearranging formula: MSE = Bias^2 + Var
                    cvLoss.([pairs{i,2} 'TemporalVar']) = ...
                        cvLoss.([pairs{i,2} 'TemporalMSE']) ...
                                        - cvLoss.([pairs{i,2} 'TemporalBias']).^2;
                
                end
            
            end
        
        
        
        end
        
        
        function aggrCorr = calcCVCorrelations( subModels, set )
            % Average correlations from submodels
            arguments
                subModels       cell
                set             char ...
                    {mustBeMember( set, {'Training', 'Validation'} )}
            end
        
            nModels = length( subModels );
            fields = fieldnames( subModels{1}.Correlations.(set) );
            nFields = length( fields );
        
            for i = 1:nFields
        
                fldDim = size( subModels{1}.Correlations.(set).(fields{i}) );
                R = zeros( fldDim );
                for k = 1:nModels
                   R = R + subModels{k}.Correlations.(set).(fields{i});
                end
                aggrCorr.(fields{i}) = R/nModels;
        
            end
        
        end
        
        
        function P = calcCVParameter( subModels, param )
            % Average a given parameter from submodels
            arguments
                subModels           cell
                param               char
            end
        
            P = 0;
            % check if parameter exists
            if ~isprop( subModels{1}, param )
                warning(['SubModel parameter ' param ' does not exist.']);
                return
            end
            if ~isnumeric( subModels{1}.(param) )
                warning(['SubModel parameter ' param ' is not numeric.']);
                return
            end
        
            fldDim = size( subModels{1}.(param) );
            P = zeros( fldDim );
            nModels = length( subModels );
            for k = 1:nModels
               P = P + subModels{k}.(param);
            end
            P = P/nModels;
        
        end
        
        
        function P = calcCVNestedParameter( subModels, param )
            % Average a given nested parameter from submodels
            arguments
                subModels           cell
                param               cell
            end
        
            P = 0;
            % check if parameter exists
            try
                fld = getfield( subModels{1}, param{:} );
            catch
                warning('SubModel parameter hierarchy does not exist.');
                return
            end
            
            if ~isnumeric(fld)
                warning(['SubModel parameter ' param ' is not numeric.']);
                return
            end
        
            fldDim = size(fld);
            P = zeros( fldDim );
            nModels = length( subModels );
            for k = 1:nModels
               P = P + getfield( subModels{k}, param{:} );
            end
            P = P/nModels;
        
        end

    end

end

