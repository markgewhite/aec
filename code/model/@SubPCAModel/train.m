function self = train( self, thisData )
    % Run FPCA for the encoder
    arguments
        self         SubPCAModel
        thisData     ModelDataset
    end

    % create a functional data object with fewer bases
    XFd = smooth_basis( self.TSpan.Regular, ...
                        thisData.XInputRegular, ...
                        self.FDA.FdParamsRegular );

    pcaStruct = pca_fd( XFd, self.ZDim );

    self.MeanFd = pcaStruct.meanfd;
    self.CompFd = pcaStruct.harmfd;
    self.VarProp = pcaStruct.varprop;

    if size( pcaStruct.harmscr, 3 ) == 1
        pcaStruct.harmscr = permute( pcaStruct.harmscr, [1 3 2] );
    end
    self.ZStd = squeeze( std(pcaStruct.harmscr) );

    % generate the latent components
    Z = reshape( pcaStruct.harmscr, size(pcaStruct.harmscr, 1), [] );

    % standardize
    self.AuxModelZMean = mean( Z );
    self.AuxModelZStd = std( Z );
    Z = (Z-self.AuxModelZMean)./self.AuxModelZStd;

    % train the auxiliary model
    switch self.AuxModelType
        case 'Logistic'
            self.AuxModel = fitclinear( Z, thisData.Y, ...
                                        Learner = "logistic");
        case 'Fisher'
            self.AuxModel = fitcdiscr( Z, thisData.Y );
        case 'SVM'
            self.AuxModel = fitcecoc( Z, thisData.Y );
    end

    % compute the mean curve directly
    self.MeanCurve = eval_fd( self.TSpan.Regular, self.MeanFd );

    % compute the components' explained variance
    [self.AuxModelALE, self.ALEQuantiles, ...
        self.LatentComponents ] = self.getLatentResponse( thisData );

    % set the oversmoothing level
    XHat = self.reconstruct( Z );

    [ self.FDA.FdParamsTarget, self.FDA.LambdaTarget ] = ...
        thisData.setFDAParameters( thisData.TSpan.Target, ...
                                   permute(XHat, [1 3 2]) );
    
    % plot them on specified axes
    plotLatentComp( self, type = 'Smoothed', shading = true );

    % plot the Z distributions
    plotZDist( self, Z );

    % plot the Z clusters
    plotZClusters( self, Z, Y = thisData.Y );

end
