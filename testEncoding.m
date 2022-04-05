% ************************************************************************
% Script: test encoding of synthetic data
%
% ************************************************************************

clear;

rng( 0 );

nCodes = 4;
nRuns = 100;
nPts = 101; % 21 for JumpVGRF
dataSource = 'JumpVGRF';

errAE = zeros( nRuns, 1 );
errPCA = zeros( nRuns, 1 );
errNet = zeros( nRuns, 1 );
for i = 1:nRuns

    disp(['*** Iteration = ' num2str(i) ' ***']);

    % prepare data
    [X, XN, XFd, Y, setup.data ] = initializeData( dataSource, nCodes, nPts ); 

    % partitioning
    cvPart = cvpartition( Y, 'Holdout', 0.5 );
    XTrn = splitData( X, training(cvPart) );
    XTst = splitData( X, test(cvPart) );
    XNTrn = splitData( XN, training(cvPart) );
    XNTst = splitData( XN, test(cvPart) );
    YTrn = Y( training(cvPart) );
    YTst = Y( test(cvPart)  );

    if setup.data.embedding
        % generate embedding with transform
        [XTrn, XTst, setup.data.embed.params ] = ...
                    genEmbedding( XTrn, XTst, setup.data.embed );
        setup.data.nFeatures = size( XTrn, 1 );
    end
    

    disp('Generated and partitioned data.');

    % initialise plots
    ax = initializePlots( nCodes, setup.data.nChannels );

    % ----- autoencoder -----

    % initalise autoencoder setup
    setup.aae = initializeAE( setup.data );

    % train the autoencoder
    [dlnetEnc, dlnetDec, dlnetDis, dlnetCls] = ...
                    trainAAE( XTrn, XNTrn, YTrn, setup.aae, ax );

    % switch to DL array format
    dlXTrn = dlarray( XTrn, 'CB' );
    dlXTst = dlarray( XTst, 'CB' );

    % generate encodings
    dlZTrn = getEncoding( dlnetEnc, dlXTrn, setup.aae );
    dlZTst = getEncoding( dlnetEnc, dlXTst, setup.aae );

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
    dlXNTrnHat = predict( dlnetDec, dlZTrn );
    dlXNTstHat = predict( dlnetDec, dlZTst );
    XNTrnHat = double(extractdata( dlXNTrnHat ));
    XNTstHat = double(extractdata( dlXNTstHat ));
    if size( XNTrnHat, 3 ) > 1
        XNTrnHat = permute( XNTrnHat, [1 3 2] );
        XNTstHat = permute( XNTstHat, [1 3 2] );
    end

    errTrn = sqrt( mse( XNTrn, XNTrnHat ) );
    disp( ['AE Training Error = ' num2str(errTrn)] );
    errTst = sqrt( mse( XNTst, XNTstHat ) );
    disp( ['AE Testing Error  = ' num2str(errTst)] );

    % plot resulting curves
    XTstHatFd = smooth_basis( setup.data.fda.tSpan, XNTstHat, ...
                                setup.data.fda.fdPar );

    %subplotFd( ax.ae.pred, XTstHatFd );
    %title( ax.ae.pred, 'AE Prediction' );



    % ----- PCA encoding -----

    disp('Running PCA ... ');
    XTrnFd = smooth_basis( setup.data.fda.tSpan, XNTrn, setup.data.fda.fdPar );
    XTstFd = smooth_basis( setup.data.fda.tSpan, XNTst, setup.data.fda.fdPar );

    % setup the PCA baseline model
    baselineModel = pcaModel( setup.data.fda.fdPar, nFeatures = nCodes );
    baselineModel = train( baselineModel, XTrnFd );

    % generate latent codes 
    ZTrnPCA = encode( baselineModel, XTrnFd );
    ZTstPCA = encode( baselineModel, XTstFd );
    
    % reconstruct the curves from those codes
    XTrnFdPCA = reconstruct( baselineModel, ZTrnPCA );
    XTstFdPCA = reconstruct( baselineModel, ZTstPCA );

    XTrnHatPCA = eval_fd( setup.data.fda.tSpan, XTrnFdPCA );
    XTstHatPCA = eval_fd( setup.data.fda.tSpan, XTstFdPCA );
    
    errTrnPCA = sqrt( mse( XNTrn, XTrnHatPCA ) );
    disp( ['PCA Training Error = ' num2str(errTrnPCA)] );
    errTstPCA = sqrt( mse( XNTst, XTstHatPCA ) );
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




