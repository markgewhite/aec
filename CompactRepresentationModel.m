classdef CompactRepresentationModel
    % Super class encompassing all individual dimensional reduction models

    properties
        XInputDim       % X dimension (number of points) for input
        XTargetDim      % X dimension for output
        ZDim            % Z dimension (number of features)
        CDim            % C dimension (number of classes)
        XChannels       % number of channels in X
        Scale           % scaling factor for reconstruction loss
        AuxModelType    % type of auxiliary model to use
        AuxModel        % auxiliary model

        ShowPlots       % flag whether to show plots
        Figs            % figures holding the plots
        Axes            % axes for plotting latent space and components
        NumCompLines    % number of lines in the component plot

        LatentComponents % computed components across partitions
        ComponentVar     % component variance of each ompoent
        VarProportion    % proportion total variance explained by each component

        Predictions      % training and validation predictions
        Loss             % training and validation losses
    end

    methods

        function self = CompactRepresentationModel( theFullModel )
            % Initialize the model
            arguments
                theFullModel        FullRepresentationModel
            end

            self.XInputDim = theFullModel.XInputDim;
            self.XTargetDim = theFullModel.XTargetDim;
            self.ZDim = theFullModel.ZDim;
            self.CDim = theFullModel.CDim;
            self.XChannels = theFullModel.XChannels;
            self.Scale = theFullModel.Scale;
            self.AuxModelType = theFullModel.AuxModelType;
            self.ShowPlots = theFullModel.ShowPlots;
            self.Figs = theFullModel.Figs;
            self.Axes = theFullModel.Axes;
            self.NumCompLines = theFullModel.NumCompLines;

        end


        function self = evaluate( self, thisTrnSet, thisValSet )
            % Evaluate the model with a specified dataset
            % It may be a full or compact model
            arguments
                self            CompactRepresentationModel
                thisTrnSet      modelDataset
                thisValSet      modelDataset
            end

            [ self.Loss.Training, self.Predictions.Training ] = ...
                                evaluateDataset( self, thisTrnSet );

            [ self.Loss.Validation, self.Predictions.Validation ] = ...
                                evaluateDataset( self, thisValSet );

        end

    end

    methods (Static)


        function err = getReconLoss( self, X, XHat )
            % Compute the reconstruction loss
            arguments
                self        CompactRepresentationModel
                X           double
                XHat        double
            end

            err = mean( (XHat-X).^2, 'all' );
        
        end


        function [ XCReg, varProp, compVar ] = ...
                                    genLatentComponents( self, thisDataset )
            % Generate the latent components and
            % compute the explained variance
            arguments
                self            CompactRepresentationModel            
                thisDataset     modelDataset
            end

            % generate the latent encodings
            Z = self.encode( thisDataset );

            % generate the AE components, smoothing them, for storage
            XC = self.latentComponents( Z, ...
                                        sampling = 'Fixed', ...
                                        convert = true );

            XCFd = smooth_basis( thisDataset.TSpan.Target, ...
                                 XC, ...
                                 thisDataset.FDA.FdParamsTarget );
            XCReg = squeeze( ...
                        eval_fd( thisDataset.TSpan.Regular, XCFd ) );

            % generate finer-grained components for calculation
            [ XCFine, offsets ] = self.latentComponents( ...
                                            Z, ...
                                            sampling = 'Fixed', ...
                                            nSample = 100, ...
                                            centre = false, ...
                                            convert = true );
    
            XCFineFd = smooth_basis( thisDataset.TSpan.Target, ...
                                 XCFine, ...
                                 thisDataset.FDA.FdParamsTarget );
            XCFineReg = squeeze( ...
                        eval_fd( thisDataset.TSpan.Regular, XCFineFd ) );

            % get the input data in a smoothed form
            XReg = squeeze( ...
                        eval_fd( thisDataset.TSpan.Regular, ...
                                 thisDataset.XFd ) );

            % compute the components' explained variance
            [varProp, compVar] = self.explainedVariance( ...
                                        self, XReg, XCFineReg, offsets );    

        end


        function [ varProp, compVar ] = explainedVariance( self, X, XC, offsets )
            % Compute the explained variance for the components
            arguments
                self            CompactRepresentationModel
                X
                XC
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
            if size( XC, 3 ) > 1
                X = permute( X, [1 3 2] );
                XC = permute( XC, [1 3 2] );
            end

            if mod( size( XC, 2 ), 2 )==1
                % remove the XC mean curve at the end
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


    methods (Abstract)

        % Train the model on the data provided
        self = train( self, thisDataset )

        % Encode features Z from X using the model - placeholder
        Z = encode( self, X )

        % Reconstruct X from Z using the model - placeholder
        XHat = reconstruct( self, Z )

    end

end


function [ eval, pred ] = evaluateDataset( self, thisDataset )
    % Evaluate the model with a specified dataset
    arguments
        self             CompactRepresentationModel
        thisDataset      modelDataset
    end

    % record the input
    pred.XTarget = squeeze( thisDataset.XTarget );
    pred.XRegular = squeeze( thisDataset.XInputRegular );
    pred.Y = thisDataset.Y;

    % generate latent encoding using the trained model
    pred.Z = self.encode( thisDataset );

    % reconstruct the curves
    pred.XHat = squeeze( self.reconstruct( pred.Z ) );

    % smooth the reconstructed curves
    XHatFd = smooth_basis( thisDataset.TSpan.Target, ...
                           pred.XHat, ...
                           thisDataset.FDA.FdParamsTarget );
    pred.XHatSmoothed = squeeze( ...
                eval_fd( thisDataset.TSpan.Target, XHatFd ) );
    
    pred.XHatRegular = squeeze( ...
                eval_fd( thisDataset.TSpan.Regular, XHatFd ) );

    % compute reconstruction loss
    eval.ReconLoss = self.getReconLoss( thisDataset.XTarget, pred.XHat );
    eval.ReconLossSmoothed = self.getReconLoss( pred.XHatSmoothed, pred.XHat );

    % compute reconstruction loss for the regularised curves
    eval.ReconLossRegular = self.getReconLoss( pred.XHatRegular, pred.XRegular );

    % compute the mean squared error as a function of time
    eval.ReconTimeMSE = self.getReconTemporalLoss( pred.XHatRegular, pred.XRegular );

    % compute the auxiliary loss using the model
    ZLong = reshape( pred.Z, size( pred.Z, 1 ), [] );
    pred.AuxModelYHat = predict( self.AuxModel, ZLong );
    eval.AuxModelLoss = getPropCorrect( pred.AuxModelYHat, pred.Y );

    if isa( self, 'autoencoderModel' )
        
        % compute the comparator loss using the comparator network
        [ pred.ComparatorYHat, eval.ComparatorLoss ] = ...
                    predictComparator( ...
                            self, ...
                            thisDataset.getDLInput( self.XDimLabels ), ...
                            thisDataset.Y );

        % compute the auxiliary loss using the network
        [ pred.AuxNetworkYHat, eval.AuxNetworkLoss ] = ...
                    predictAux( self, pred.Z, thisDataset.Y );

    else
        eval.ComparatorLoss = [];
        eval.AuxNetworkLoss = [];

    end

end
