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

        ShowPlots       % flag whether to show plots
        Figs            % figures holding the plots
        Axes            % axes for plotting latent space and components
        NumCompLines    % number of lines in the component plot

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


        function [ ZC, offsets, nObs ] = componentEncodings( self, Z, args )
            % Calculate the funtional components from the latent codes
            % using the decoder network. For each component, the relevant 
            % code is varied randomly about the mean. This is more 
            % efficient than calculating two components at strict
            % 2SD separation from the mean.
            arguments
                self                CompactRepresentationModel
                Z                   {mustBeA( Z, {'dlarray', 'double'} )}
                args.sampling       char ...
                                    {mustBeMember(args.sampling, ...
                                        {'Random', 'Fixed'} )} = 'Random'
                args.nSample        double {mustBeInteger} = 0
                args.range          double {mustBePositive} = 2.0
                args.ZSizeMax       double = 100
            end

            if args.nSample > 0
                nSample = args.nSample;
            else
                nSample = self.NumCompLines;
            end

            nObs = size( Z, 2 );
            
            % generate the Z offset factors about the mean
            switch args.sampling
                case 'Random'
                    % generate centred uniform distribution
                    offsets = 2*args.range*(rand( nSample, 1 )-0.5);
        
                case 'Fixed'
                    offsets = linspace( -args.range, args.range, nSample );
            end

            % convert the z-scores (offsets) to percentiles
            % giving a preponderance of values at the tails
            prc = 100*normcdf( offsets );


            switch self.ComponentType

                case 'Mean'
                    % initialise the components' Z codes at the mean
                    % include an extra one that will be preserved
                    ZMean = mean( Z, 2 );                    
                    ZC = repmat( ZMean, 1, self.ZDim*nSample+1 );
                    nRepeats = 1;

                case 'PDP'
                    % initialize with all Z codes duplicated
                    % but limit the total size
                    if nObs > args.ZSizeMax
                        Z = Z( :, randsample( nObs, args.ZSizeMax ) );
                        nObs = args.ZSizeMax;
                    end
                    ZC = repmat( Z, 1, self.ZDim*nSample+1 );
                    nRepeats = size( Z, 2 );

            end

            if isa( Z, 'dlarray' )
                % convert to double for speed
                Z = double(extractdata( Z ));
            end

            for j = 1:nSample
                
                for i =1:self.ZDim
                    colStart = ((i-1)*nSample+j-1)*nRepeats+1;
                    colEnd = colStart + nRepeats - 1;
                    ZC( i, colStart:colEnd ) = prctile( Z(i,:), prc(j) );
                end

            end

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


        function self = clearGraphics( self )
            % Clear the graphics objects to save memory
            arguments
                self            CompactRepresentationModel
            end

            self.Figs = [];
            self.Axes = [];

        end


        function [ XCReg, varProp, compVar ] = ...
                                    getLatentComponents( self, thisDataset )
            % Generate the latent components and
            % compute the explained variance
            arguments
                self            CompactRepresentationModel            
                thisDataset     ModelDataset
            end

            % generate the latent encodings
            Z = self.encode( thisDataset );

            % generate the AE components, smoothing them, for storage
            XC = self.latentComponents( Z, ...
                                        sampling = 'Fixed', ...
                                        centre = false, ...
                                        convert = true );

            XCReg = smoothSeries( XC, ...
                                  self.TSpan.Target, ...
                                  self.TSpan.Regular, ...
                                  self.FDA.FdParamsTarget );

            % generate finer-grained components for calculation
            [ XCFine, offsets ] = self.latentComponents( ...
                                            Z, ...
                                            sampling = 'Fixed', ...
                                            nSample = 10, ...
                                            centre = false, ...
                                            convert = true );
    
            XCFineReg = smoothSeries( XCFine, ...
                                  thisDataset.TSpan.Target, ...
                                  thisDataset.TSpan.Regular, ...
                                  thisDataset.FDA.FdParamsTarget );

            % compute the components' explained variance
            [varProp, compVar] = self.explainedVariance( ...
                                thisDataset.XTarget, XCFineReg, offsets );    

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
            % !!! Is there a problem here? !!!
            if size( XC, 3 ) > 1
                X = permute( X, [1 3 2] );
                XC = permute( XC, [1 3 2] );
            end

            if mod( size( XC, 2 ), 2 )==1
                % remove the XC mean curve at the end
                % !!! or is there a problem here? !!!
                if size( XC, 3 ) > 1
                    XC = XC( :, :, 1:end-1 );
                else
                    XC = XC( :, 1:end-1 );
                end
            end

            % centre using the X mean curve (XC mean is almost identical)
            X = X - mean( X, 2 );
            XC = XC - mean( XC, 2 );
            
            % compute the total variance from X
            totVar = mean( sum( X.^2 ) );

            % reshape XC by introducing dim for offset
            nOffsets = length( offsets );
            XC = reshape( XC, size(XC,1), self.XChannels, nOffsets, self.ZDim );

            % compute the component variances in turn
            compVar = zeros( self.XChannels, nOffsets, self.ZDim );

            for i = 1:self.XChannels
                for j = 1:nOffsets
                    for k = 1:self.ZDim
                        compVar( i, j, k ) = sum( (XC(:,i,j,k)/offsets(j)).^2 );
                    end
                end
            end

            compVar = squeeze( compVar./totVar );
            varProp = mean( compVar );

        end


    end


    methods (Static)

        function [loss, pred, cor] = evaluateSet( thisModel, thisDataset )
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
            pred.XHat = squeeze( thisModel.reconstruct( pred.Z ) );
        
            % smooth the reconstructed curves
            pred.XHatSmoothed = smoothSeries( pred.XHat, ...
                                  thisDataset.TSpan.Target, ...
                                  thisDataset.TSpan.Target, ...
                                  thisDataset.FDA.FdParamsTarget );

            pred.XHatRegular = smoothSeries( pred.XHat, ...
                                  thisDataset.TSpan.Target, ...
                                  thisDataset.TSpan.Regular, ...
                                  thisDataset.FDA.FdParamsTarget );
        
            % compute reconstruction loss
            loss.ReconLoss = reconLoss( thisDataset.XTarget, pred.XHat, ...
                                        thisModel.Scale );
            loss.ReconLossSmoothed = reconLoss( pred.XHatSmoothed, pred.XHat, ...
                                                thisModel.Scale );
        
            % compute reconstruction loss for the regularised curves
            loss.ReconLossRegular = reconLoss( pred.XHatRegular, pred.XRegular, ...
                                               thisModel.Scale );
        
            % compute the auxiliary loss using the model
            ZLong = reshape( pred.Z, size( pred.Z, 1 ), [] );
            pred.AuxModelYHat = predict( thisModel.AuxModel, ZLong );
            loss.AuxModelLoss = getPropCorrect( pred.AuxModelYHat, pred.Y );
               
            
            % compute the mean squared error as a function of time
            loss.ReconTimeMSE = reconTemporalLoss( pred.XHat, pred.XTarget, ...
                                                   thisModel.Scale );
        
            % compute the mean error (bias) as a function of time
            loss.ReconTimeBias = reconTemporalBias( pred.XHat, pred.XTarget, ...
                                                   thisModel.Scale );
        
            % compute the variance as a function of time
            loss.ReconTimeVar = reconTemporalLoss( ...
                            pred.XHat - loss.ReconTimeBias, ...
                            pred.XTarget, thisModel.Scale );
        
            % compute the mean squared error as a function of time
            loss.ReconTimeMSERegular = reconTemporalLoss( pred.XHatRegular, pred.XRegular, ...
                                                   thisModel.Scale );
        
            % compute the mean error (bias) as a function of time
            loss.ReconTimeBiasRegular = reconTemporalBias( pred.XHatRegular, pred.XRegular, ...
                                                   thisModel.Scale );
        
            % compute the variance as a function of time
            loss.ReconTimeVarRegular = reconTemporalLoss( ...
                            pred.XHatRegular - loss.ReconTimeBiasRegular, ...
                            pred.XRegular, thisModel.Scale );

            % compute the latent code correlation matrix
            [ cor.ZCorrelation, cor.ZCovariance ] = ...
                latentCodeCorrelation( pred.Z, summary = true );
            
            [ cor.ZCorrelationMatrix, cor.ZCovarianceMatrix ] = ...
                latentCodeCorrelation( pred.Z );

            % compute the latent component correlation matrix
            [ cor.XCCorrelation, cor.XCCovariance ] = ...
                latentComponentCorrelation( ...
                    thisModel.LatentComponents, thisModel.NumCompLines, ...
                    summary = true );

            [ cor.XCCorrelationMatrix, cor.XCCovarianceMatrix ] = ...
                latentComponentCorrelation( ...
                    thisModel.LatentComponents, thisModel.NumCompLines );
        
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

