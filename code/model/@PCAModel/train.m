function self = train( self, thisData )
    % Run FPCA for the encoder
    arguments
        self         PCAModel
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
    ZAux = Z( :, 1:self.ZDimAux );
    self.AuxModelZMean = mean( ZAux );
    self.AuxModelZStd = std( ZAux );
    Z = (Z-self.AuxModelZMean)./self.AuxModelZStd;

    % train the auxiliary model
    switch self.AuxModelType
        case 'Logistic'
            self.AuxModel = fitclinear( ZAux, thisData.Y, ...
                                        Learner = "logistic");
        case 'Fisher'
            self.AuxModel = fitcdiscr( ZAux, thisData.Y );
        case 'SVM'
            self.AuxModel = fitcecoc( ZAux, thisData.Y );
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
    self.plotLatentComp( type = 'Smoothed', shading = true );

    % plot the Z distributions
    self.plotZDist( Z );

    % plot the Z clusters
    self.plotZClusters( Z, Y = thisData.Y );

end
