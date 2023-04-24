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
    self.ZStd = squeeze( std(pcaStruct.harmscr) );

    % separate out the portion of Z for the auxiliary model
    Z = reshape( pcaStruct.harmscr, size(pcaStruct.harmscr, 1), [] );
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
    self.MeanCurve = eval_fd( self.TSpan.Regular, self.MeanFd );

    % get the auxiliary model's response to each Z element
    self = self.getAuxResponse( thisData );

    % compute the functional components
    self.LatentResponseFcn = @(Z) self.reconstruct( Z' );
    self.LatentComponents = self.getLatentResponse( thisData );

    % set the oversmoothing level
    XHat = self.reconstruct( Z );

    [ self.FDA.FdParamsTarget, self.FDA.LambdaTarget ] = ...
        thisData.setFDAParameters( thisData.TSpan.Target, ...
                                   permute(XHat, [1 3 2]) );

end
