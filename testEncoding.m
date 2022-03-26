% ************************************************************************
% Script: test encoding of synthetic data
%
% ************************************************************************

clear;

rng( 0 );

nCodes = 4;
nRuns = 100;
nPts = 101; % 21 for JumpVGRF
nPtsFine = 101; % 201 for JumpVGRF
dataSource = 'JumpVGRF';

errAE = zeros( nRuns, 1 );
errPCA = zeros( nRuns, 1 );
errNet = zeros( nRuns, 1 );
for i = 1:nRuns

    disp(['*** Iteration = ' num2str(i) ' ***']);

    % prepare data
    [XIn, XGen, XFd, Y, setup.data ] = initializeData( dataSource, nCodes, ...
                                               nPts, nPtsFine ); 

    % partitioning
    cvPart = cvpartition( Y, 'Holdout', 0.5 );
    XTrn = splitData( XIn, training(cvPart) );
    XTst = splitData( XIn, test(cvPart) );
    XGTrn = splitData( XGen, training(cvPart) );
    XGTst = splitData( XGen, test(cvPart) );
    YTrn = Y( training(cvPart) );
    YTst = Y( test(cvPart)  );

    if setup.data.embedding
        % generate embedding with transform
        [XTTrn, XTTst, setup.data.embed.params ] = ...
                    genEmbedding( XTrn, XTst, setup.data.embed );
    else
        % flatten the inputs in 'CB' dimensions
        XTTrn = reshape( XGTrn, size(XGTrn,1)*size(XGTrn,3), size(XGTrn,2) );
        XTTst = reshape( XGTst, size(XGTst,1)*size(XGTst,3), size(XGTst,2) );
    end
    setup.data.nFeatures = size( XTTrn, 1 );

    disp('Generated and partitioned data.');

    % initialise plots
    ax = initializePlots( nCodes, setup.data.nChannels );

    % ----- autoencoder -----

    % initalise autoencoder setup
    setup.aae = initializeAE( setup.data );

    % train the autoencoder
    [dlnetEnc, dlnetDec, dlnetDis, dlnetCls] = ...
                    trainAAE( XGTrn, XTTrn, YTrn, setup.aae, ax );

    % switch to DL array format
    dlXTTrn = dlarray( XTTrn, 'CB' );
    dlXTTst = dlarray( XTTst, 'CB' );

    % generate encodings
    dlZTrn = getEncoding( dlnetEnc, dlXTTrn, setup.aae );
    dlZTst = getEncoding( dlnetEnc, dlXTTst, setup.aae );

    % convert back to numeric arrays
    ZTrn = double(extractdata( dlZTrn ));
    ZTst = double(extractdata( dlZTst ));

    % plot Z distribution 
    plotZDist( ax.ae.distZTrn, ZTrn, 'AE: Z Train', true );
    plotZDist( ax.ae.distZTst, ZTst, 'AE: Z Test', true );

    % plot characteristic features
    for j = 1:setup.data.nChannels
        plotLatentComp( ax.ae.comp(:,j), dlnetDec, ZTrn, j, ...
                    setup.data.fda.tSpan, setup.data.fda.fdPar );
    end

    % classify using discriminant analysis
    model = fitcdiscr( ZTrn', YTrn );
    errAE(i) = loss( model, ZTst', YTst );
    disp( ['FITCDISCR:          Holdout Loss = ' ...
                            num2str( errAE(i), '%0.3f') ]);

    % classify using the trained network
    dlYHatTst = predict( dlnetCls, dlZTst );
    YHatTst = double(   ...
            onehotdecode( dlYHatTst, single(setup.aae.cLabels), 1 ) )' - 1;
    errNet(i) = sum( YHatTst~=YTst )/length(YTst);
    disp( ['NETWORK CLASSIFIER: Holdout Loss = ' ...
                            num2str( errNet(i), '%0.3f') ]);

    % plot the clusters for test data using the training loadings 
    % from a canonical discriminant analysis
    ZTrnNse = ZTrn + 1E-6*randn(size(ZTrn)); % to avoid errors
    ZTrnCanInfo = cda( ZTrn', YTrn );
    ZTstCan = ZTst'*ZTrnCanInfo.loadings;
    if size( ZTstCan, 2 )==1
        % only one canonical dimension
        ZTstCan = [ ZTstCan ZTstCan ]; %#ok<AGROW> 
    end
    plotClusters( ax.ae.cls, ZTstCan, YTst, YHatTst );
    title( ax.ae.cls, 'AE Encoding' );
    drawnow;

    % reconstruct the curves and calculate errors
    dlXGTrnHat = predict( dlnetDec, dlZTrn );
    dlXGTstHat = predict( dlnetDec, dlZTst );
    XGTrnHat = double(extractdata( dlXGTrnHat ));
    XGTstHat = double(extractdata( dlXGTstHat ));
    if size( XGTrnHat, 3 ) > 1
        XGTrnHat = permute( XGTrnHat, [1 3 2] );
        XGTstHat = permute( XGTstHat, [1 3 2] );
    end

    errTrn = sqrt( mse( XGTrn, XGTrnHat ) );
    disp( ['AE Training Error = ' num2str(errTrn)] );
    errTst = sqrt( mse( XGTst, XGTstHat ) );
    disp( ['AE Testing Error  = ' num2str(errTst)] );

    % plot resulting curves
    XTstHatFd = smooth_basis( setup.data.fda.tSpan, XGTstHat, ...
                                setup.data.fda.fdPar );

    %subplotFd( ax.ae.pred, XTstHatFd );
    %title( ax.ae.pred, 'AE Prediction' );



    % ----- PCA encoding -----

    disp('Running PCA ... ');
    XTrnFd = smooth_basis( setup.data.fda.tSpan, XGTrn, setup.data.fda.fdPar );
    XTstFd = smooth_basis( setup.data.fda.tSpan, XGTst, setup.data.fda.fdPar );
    pcaXTrnFd = pca_fd( XTrnFd, nCodes );

    % generate predictions and calculate errors
    ZTrnPCA = pcaXTrnFd.harmscr;
    ZTstPCA = pca_fd_score( XTstFd, ...
                          pcaXTrnFd.meanfd, ...
                          pcaXTrnFd.harmfd, ...
                          nCodes, true );
    
    XTrnFdPCA = pcaXTrnFd.meanfd + pcaXTrnFd.fdhatfd;
    XTstFdPCA = reconstructFd( pcaXTrnFd, ZTstPCA, setup.data.fda );

    XTrnHatPCA = eval_fd( setup.data.fda.tSpan, XTrnFdPCA );
    XTstHatPCA = eval_fd( setup.data.fda.tSpan, XTstFdPCA );
    
    errTrnPCA = sqrt( mse( XGTrn, XTrnHatPCA ) );
    disp( ['PCA Training Error = ' num2str(errTrnPCA)] );
    errTstPCA = sqrt( mse( XGTst, XTstHatPCA ) );
    disp( ['PCA Testing Error  = ' num2str(errTstPCA)] );

    %subplotFd( ax.pca.pred, XTstFdPCA );
    %title( ax.pca.pred, 'PCA Prediction');

    % plot Z distribution
    ZTrnPCA = reshape( ZTrnPCA, size(ZTrnPCA,1), ...
                            size(ZTrnPCA,2)*size(ZTrnPCA,3) );
    ZTstPCA = reshape( ZTstPCA, size(ZTstPCA,1), ...
                            size(ZTstPCA,2)*size(ZTstPCA,3) );
    plotZDist( ax.pca.distZTrn, ZTrnPCA', 'PCA: Z Train', true );
    plotZDist( ax.pca.distZTst, ZTstPCA', 'PCA: Z Test', true );

    % classify
    modelPCA = fitcdiscr( ZTrnPCA, YTrn );
    YHatTstPCA = predict( modelPCA, ZTstPCA );
    errPCA(i) = loss( modelPCA, ZTstPCA, YTst );
    disp( ['FITCDISCR:          Holdout Loss = ' ...
                            num2str( errPCA(i), '%0.3f') ]);

    % plot the clusters
    plotClusters( ax.pca.cls, ZTstPCA, YTst, YHatTstPCA );
    title( ax.pca.cls, 'PCA Encoding' );
    drawnow;

    pause;

end

disp( ['Mean Disciminant Classification Error  = ' ...
                num2str(mean(errAE), '%.3f') ' +/- ' ...
                num2str(std(errAE), '%.3f')] );
disp( ['Mean Network Classification Error  = ' ...
                num2str(mean(errNet), '%.3f') ' +/- ' ...
                num2str(std(errNet), '%.3f')] );
disp( ['Mean PCA Classification Error = ' ...
                num2str(mean(errPCA), '%.3f') ' +/- ' ...
                num2str(std(errPCA), '%.3f')] );
disp( ['Frequency AE error is lower than PCA error = ' ...
            num2str( sum(errAE<errPCA) ) ] );
disp( ['Frequency Network error is lower than AE error = ' ...
            num2str( sum(errNet<errAE) ) ] );




