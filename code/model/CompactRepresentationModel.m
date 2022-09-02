classdef CompactRepresentationModel
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

        ShowPlots       % flag whether to show plots
        Figs            % figures holding the plots
        Axes            % axes for plotting latent space and components
        NumCompLines    % number of lines in the component plot

        MeanCurve       % estimated mean curve
        ComponentType   % type of components generated (Mean or PDP)
        LatentComponents % computed components across partitions
        
        ComponentVar     % component variance of each ompoent
        VarProportion    % proportion total variance explained by each component

        Predictions      % training and validation predictions
        Loss             % training and validation losses
        Correlations     % training and validation correlations
    end

    methods

        function self = CompactRepresentationModel( theFullModel, fold )
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
            self.ComponentType = theFullModel.ComponentType;
            self.NumCompLines = theFullModel.NumCompLines;

            self.Info.Name = strcat( self.Info.Name, "-Fold", ...
                                     num2str( fold, "%02d" ) );

        end


        function [F, ZQMid, offsets ] = ALE( self, dlZ, args )
            % Accumulated Local Estimation 
            % For latent component generation and the auxiliary model
            arguments
                self                CompactRepresentationModel
                dlZ                 {mustBeA( dlZ, {'dlarray', 'double'} )}
                args.sampling       char ...
                                    {mustBeMember(args.sampling, ...
                                    {'Regular', 'Component'} )} = 'Regular'
                args.nSample        double {mustBeInteger} = 20
                args.maxObs         double = 1000
                args.modelFcn       function_handle
                args.modelFcnArgs   cell = []
            end
            
            if isa( dlZ, 'dlarray' )
                % convert to double for quantiles, sort and other functions
                Z = double(extractdata( dlZ ));
            else
                if size(dlZ,1) ~= self.ZDim
                    % transpose into standard dimensions:
                    % 1st=ZDim and 2nd=observations
                    dlZ = dlZ';
                end
                Z = dlZ;
            end

            nObs = size( dlZ, 2 );
            if nObs > args.maxObs
                % data too large - subsample
                subset = randsample( nObs, args.maxObs );
                dlZ = dlZ( :, subset );
                Z = Z( :, subset );
                nObs = args.maxObs;
            end

            % generate the quantiles and required Z values
            K = args.nSample;
            switch args.sampling
                case 'Regular'
                    prc = linspace( 1/K, 1, K );
                case 'Component'
                    prc = [   0.100, ...
                              0.225, 0.325, ...
                              0.450, 0.550, ...
                              0.675, 0.775, ...
                              0.900, 1.000];
            end
            offsets = norminv( prc ); % z-scores
            ZQ = quantile( Z, prc, 2 );
            ZQ = [ min(Z,[],2) ZQ ];
            K = length(ZQ)-1;

            % identify bin assignments across dimensions
            A = zeros( self.ZDim, nObs );
            for d = 1:self.ZDim
                [~, order] = sort( Z(d,:) );
                j = 1;
                for i = order
                    if Z(d,i)<=ZQ(d,j+1)
                        A(d,i) = j;
                    else
                        j = min( j+1, K );
                        A(d,i) = j;
                    end
                end
            end

            for d = 1:self.ZDim

                % generate predictions for bin and bin+1
                dlZC1 = dlZ;
                dlZC2 = dlZ;
                dlZC1(d,:) = ZQ(d, A(d,:));
                dlZC2(d,:) = ZQ(d, A(d,:)+1);
                if isempty( args.modelFcnArgs )
                    YHat1 = args.modelFcn( self, dlZC1 );
                    YHat2 = args.modelFcn( self, dlZC2 );
                else
                    YHat1 = args.modelFcn( self, dlZC1, args.modelFcnArgs{:} );
                    YHat2 = args.modelFcn( self, dlZC2, args.modelFcnArgs{:} );
                end
                delta = YHat2 - YHat1;

                if d==1
                    % allocate arrays knowing the size of YHat
                    if size(delta,3)==1
                        FDim(1) = size(delta,1);
                        FDim(2) = 1;
                        F = zeros( self.ZDim, K, FDim(1) );
                        FBin = zeros( K, FDim(1) );
                    else
                        FDim(1) = size(delta,1);
                        FDim(2) = size(delta,2);
                        F = zeros( self.ZDim, K, FDim(1), FDim(2) );
                        FBin = zeros( K, FDim(1), FDim(2) );
                    end
                end

                % subtract the average weighted by number of occurrences
                w = histcounts(Z(d,:), unique(ZQ(d,:)))';

                % calculate means of delta grouped by bin
                % and cumulatively sum
                if FDim(2)==1
                    for i = 1:K
                        FBin(i,:) = mean( delta(:,A(d,:)==i), 2 );
                    end
                    FMeanCS = [ zeros(1,FDim(1)); cumsum(FBin) ];
                    FMeanMid = (FMeanCS(1:K,:) + FMeanCS(2:K+1,:))/2;
                    F( d,:,: ) = FMeanMid - sum(w.*FMeanMid)/sum(w);

                else
                    for i = 1:K
                        FBin(i,:,:) = mean( delta(:,:,A(d,:)==i), 3 );
                    end
                    FMeanCS = [ zeros(1,FDim(1),FDim(2)); cumsum(FBin) ];
                    FMeanMid = (FMeanCS(1:K,:,:) + FMeanCS(2:K+1,:,:))/2;
                    F( d,:,:,: ) = FMeanMid - sum(w.*FMeanMid)/sum(w);

                end

            end

            ZQMid = ((ZQ(:,1:K) + ZQ(:,2:K+1))/2)';

        end


        function [auxALE, Z, ZQ] = auxPartialDependence( self, thisDataset, args )
            % Generate the model's partial dependence to latent codes
            arguments
                self            CompactRepresentationModel
                thisDataset     ModelDataset
                args.nSample    double {mustBeInteger} = 20
                args.auxFcn     function_handle = @predictAuxModel
            end

            % generate the latent encodings
            Z = self.encode( thisDataset );

            % define the query points by z-scores
            [auxALE, ZQ] = self.ALE( Z, ...
                              sampling = 'Regular', ...
                              modelFcn = args.auxFcn, ...
                              nSample = args.nSample, ...
                              maxObs = 10000 );

        end



        function self = evaluate( self, thisTrnSet, thisValSet )
            % Evaluate the model with a specified dataset
            % It may be a full or compact model
            arguments
                self            CompactRepresentationModel
                thisTrnSet      ModelDataset
                thisValSet      ModelDataset
            end

            [ self.Loss.Training, ...
                self.Predictions.Training, ...
                    self.Correlations.Training ] = ...
                                self.evaluateSet( self, thisTrnSet );

            if thisValSet.NumObs > 0
                [ self.Loss.Validation, ...
                    self.Predictions.Validation, ...
                        self.Correlations.Validation ] = ...
                                    self.evaluateSet( self, thisValSet );
            end

        end


        function save( self )
            % Save the model plots and the object itself
            arguments
                self            CompactRepresentationModel
            end

            plotObjects = self.Axes;
            plotObjects.Components = self.Figs.Components;

            savePlots( plotObjects, self.Info.Path, self.Info.Name );

        end


        function self = compress( self, level )
            % Clear the objects to save memory
            arguments
                self            CompactRepresentationModel
                level           double ...
                    {mustBeInRange( level, 0, 3 )} = 0
            end

            if level >= 1
                self.Figs = [];
                self.Axes = [];
            end

            if level >= 2
                self.Predictions = [];
            end

        end


        function [ XA, XC, varProp, compVar ] = getLatentResponse( self, thisDataset )
            % Generate the latent components and
            % compute the explained variance
            arguments
                self            CompactRepresentationModel            
                thisDataset     ModelDataset
            end

            % calculate the auxiliary model/network dependence
            XA = self.auxPartialDependence( thisDataset );

            % generate the AE components, smoothing them, for storage
            Z = self.encode( thisDataset, convert = false );
            XC = self.calcLatentComponents( Z, smooth = true );

            % generate finer-grained components for calculation
            [ XCFine, ~, offsets ] = self.calcLatentComponents( ...
                                            Z, smooth = true );

            % compute the components' explained variance
            XTarget = permute( thisDataset.XTarget, [ 1 3 2 ] );

            [varProp, compVar] = self.explainedVariance( ...
                                XTarget, XCFine, offsets );    

        end


        function [ varProp, compVar ] = explainedVariance( self, X, XC, offsets )
            % Compute the explained variance for the components
            arguments
                self            CompactRepresentationModel
                X               {mustBeA( X, {'double', 'dlarray'} )}
                XC              {mustBeA( XC, {'double', 'dlarray'} )}
                offsets         double
            end

            % convert to double for convenience
            if isa( X, 'dlarray' )
                X = double( extractdata( X ) );
            end  
            if isa( XC, 'dlarray' )
                XC = double( extractdata( XC ) );
            end  

            % re-order the dimensions for FDA
            if size( X, 3 ) > 1
                X = permute( X, [1 3 2] );
            end
            if size( XC, 3 ) > 1
                XC = permute( XC, [1 3 2] );
            end
            
            % compute the total variance from X
            totVar = squeeze(mean( sum( X.^2 ) ))';

            % reshape XC by introducing dim for offset
            nOffsets = length( offsets );
            XC = reshape( XC, size(XC,1), nOffsets, self.ZDim, self.XChannels );

            % compute the component variances in turn
            compVar = zeros( nOffsets, self.ZDim, self.XChannels );

            for i = 1:nOffsets
                for j = 1:self.ZDim
                    for k = 1:self.XChannels
                        compVar( i, j, k ) = sum( (XC(:,i,j,k)/offsets(i)).^2 );
                    end
                end
            end

            compVar = squeeze(mean( compVar, 1 ))./totVar;
            varProp = mean( compVar, 2 )';

        end


        function [ YHat, YHatScore] = predictAuxModel( self, Z )
            % Make prediction from Z using an auxiliary model
            arguments
                self            CompactRepresentationModel
                Z               {mustBeA(Z, {'double', 'dlarray'})}
            end

            if isa( Z, 'dlarray' )
                Z = double(extractdata(Z))';
            end

            doTranspose = (size(Z,2) ~= self.ZDim);
            if doTranspose
                Z = Z';
            end

            [YHat, YHatScore] = predict( self.AuxModel, Z );

            if doTranspose
                YHat = YHat';
                YHatScore = YHatScore';
            end

        end



    end


    methods (Static)

        function [eval, pred, cor] = evaluateSet( thisModel, thisDataset )
            % Evaluate the model with a specified dataset
            arguments
                thisModel       CompactRepresentationModel
                thisDataset     ModelDataset
            end
        
            % record the input
            pred.XTarget = squeeze( thisDataset.XTarget );
            pred.XRegular = squeeze( thisDataset.XInputRegular );
            pred.Y = thisDataset.Y;
        
            % generate latent encoding using the trained model
            pred.Z = thisModel.encode( thisDataset );
        
            % reconstruct the curves
            [ pred.XHat, pred.XHatSmoothed, pred.XHatRegular ] = ...
                    thisModel.reconstruct( pred.Z, convert = true );
               
            % compute reconstruction loss
            eval.ReconLoss = reconLoss( thisDataset.XTarget, pred.XHat, ...
                                        thisModel.Scale );
            eval.ReconLossSmoothed = reconLoss( pred.XHatSmoothed, pred.XHat, ...
                                                thisModel.Scale );
        
            % compute reconstruction loss for the regularised curves
            eval.ReconLossRegular = reconLoss( pred.XHatRegular, pred.XRegular, ...
                                               thisModel.Scale );

            % compute the bias and variance
            eval.ReconBias = reconBias( thisDataset.XTarget, pred.XHat, ...
                                        thisModel.Scale );
            eval.ReconVar = eval.ReconLoss - eval.ReconBias^2;              
            
            % compute the mean squared error as a function of time
            eval.ReconTimeMSE = reconTemporalLoss( pred.XHat, pred.XTarget, ...
                                                   thisModel.Scale );
        
            % compute the mean error (bias) as a function of time
            eval.ReconTimeBias = reconTemporalBias( pred.XHat, pred.XTarget, ...
                                                   thisModel.Scale );
        
            % compute the variance as a function of time
            if length( size(pred.XHat) ) == 2
                XDiff = pred.XHat - eval.ReconTimeBias;
            else
                XDiff = pred.XHat - reshape( eval.ReconTimeBias, ...
                                            size(eval.ReconTimeBias,1), ...
                                            1, [] );
            end
            eval.ReconTimeVar = reconTemporalLoss( XDiff, pred.XTarget, ...
                                                    thisModel.Scale );
        
            % compute the mean squared error as a function of time
            eval.ReconTimeMSERegular = reconTemporalLoss( pred.XHatRegular, pred.XRegular, ...
                                                   thisModel.Scale );
        
            % compute the mean error (bias) as a function of time
            eval.ReconTimeBiasRegular = reconTemporalBias( pred.XHatRegular, pred.XRegular, ...
                                                   thisModel.Scale );
        
            % compute the variance as a function of time
            if length( size(pred.XHatRegular) ) == 2
                XDiff = pred.XHatRegular - eval.ReconTimeBiasRegular;
            else
                XDiff = pred.XHatRegular - reshape( eval.ReconTimeBiasRegular, ...
                                            size(eval.ReconTimeBiasRegular,1), ...
                                            1, [] );
            end
            eval.ReconTimeVarRegular = reconTemporalLoss( XDiff, ...
                                            pred.XRegular, thisModel.Scale );

            % compute the latent code correlation matrix
            [ cor.ZCorrelation, cor.ZCovariance ] = ...
                latentCodeCorrelation( pred.Z, summary = true );
            
            [ cor.ZCorrelationMatrix, cor.ZCovarianceMatrix ] = ...
                latentCodeCorrelation( pred.Z );

            % compute the latent component correlation matrix
            [ cor.XCCorrelation, cor.XCCovariance ] = ...
                latentComponentCorrelation( thisModel.LatentComponents,  ...
                                            summary = true );

            [ cor.XCCorrelationMatrix, cor.XCCovarianceMatrix ] = ...
                latentComponentCorrelation( thisModel.LatentComponents );

            % compute the auxiliary loss using the model
            ZLong = reshape( pred.Z, size( pred.Z, 1 ), [] );
            ZLong = (ZLong-thisModel.AuxModelZMean)./thisModel.AuxModelZStd;

            pred.AuxModelYHat = predict( thisModel.AuxModel, ZLong );
            eval.AuxModel = evaluateClassifier( pred.Y, pred.AuxModelYHat );

            % store the model coefficients - all important
            switch class( thisModel.AuxModel )
                case 'ClassificationLinear'
                    eval.AuxModel.Coeff = thisModel.AuxModel.Beta;
                case 'ClassificationDiscriminant'
                    eval.AuxModel.Coeff = thisModel.AuxModel.DeltaPredictior;
            end

            eval = flattenStruct( eval );

        end

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

