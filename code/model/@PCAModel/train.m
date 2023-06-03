function self = train( self, thisData )
    % Run FPCA for the encoder
    arguments
        self         PCAModel
        thisData     ModelDataset
    end

    % complete initialization
    [self, thisData] = self.finalizeInit( thisData );

    % perform principal components analysis
    pcaStruct = pca_fd( thisData.XFd, self.ZDim );

    self.MeanFd = pcaStruct.meanfd;
    self.CompFd = pcaStruct.harmfd;
    self.VarProp = pcaStruct.varprop;
    self.ZStd = squeeze( std(pcaStruct.harmscr) );

    % separate out the portion of Z for the auxiliary model
    Z = pcaStruct.harmscr;
    ZAux = reshape( pcaStruct.harmscr( :, 1:self.ZDimAux, :), ...
                    size(pcaStruct.harmscr, 1), [] );

    % standardize
    self.AuxModelZMean = mean( ZAux );
    self.AuxModelZStd = std( ZAux );
    ZAux = (ZAux-self.AuxModelZMean)./self.AuxModelZStd;

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
    self.MeanCurve = eval_fd( self.TSpan.Input, self.MeanFd );

    % compute the functional components
    self.LatentComponents = self.calcLatentComponents( Z );

    % set the oversmoothing level
    XHat = self.reconstruct( Z );

    [ self.FDA.FdParamsTarget, self.FDA.LambdaTarget ] = ...
        thisData.setFDAParameters( self.TSpan.Target, ...
                                   permute(XHat, [1 3 2]) );
    
    self.FDA.FdParamsComponent = self.FDA.FdParamsTarget;
    self.FDA.LambdaComponent = self.FDA.LambdaTarget;

end
