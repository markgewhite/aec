% ************************************************************************
% Script: test encoding of synthetic data
%
% ************************************************************************

clear;

N = 100;
classSizes = [ N N N ];
nDim = 1;
nCodes = 10;
nRuns = 100;

rng( 'default' );

setup.data.tFine = linspace( 0, 1000, 21 );
setup.data.tSpan = linspace( 0, 1024, 33 );
setup.data.ratio = [ 4 8 16];
setup.data.sharedLevel = 3;
setup.data.mu = [1 4 8];
setup.data.sigma = [1 6 1];
setup.data.eta = 0.1;
setup.data.warpLevel = 2;
setup.data.tau = 0;

setup.fda.basisOrder = 4;
setup.fda.penaltyOrder = 2;
setup.fda.lambda = 1E2;
setup.fda.nBasis = 20+setup.fda.penaltyOrder+1;
setup.fda.basisFd = create_bspline_basis( ...
                        [ setup.data.tSpan(1), setup.data.tSpan(end) ], ...
                          setup.fda.nBasis, setup.fda.basisOrder);
setup.fda.fdPar = fdPar( setup.fda.basisFd, ...
                         setup.fda.penaltyOrder, ...
                         setup.fda.lambda ); 

setup.reg.nIterations = 2;
setup.reg.nBasis = 12; 
setup.reg.basisOrder = 3; 
setup.reg.wLambda = 1E-2; 
setup.reg.XLambda = 1E2;
setup.reg.usePC = true;
setup.reg.nPC = 3;


errAE = zeros( nRuns, 1 );
errPCA = zeros( nRuns, 1 );
for i = 1:nRuns

    disp(['*** Iteration = ' num2str(i) ' ***']);

    % generate a new data set for each iteration
    Xraw = genSyntheticData( classSizes, nDim, setup.data );
    XFd = smooth_basis( setup.data.tSpan, Xraw, setup.fda.fdPar );
    X = eval_fd( setup.data.tFine, XFd );

    % classes
    Y = [ repelem(1,N) repelem(2,N) repelem(3,N) ]';

    % partitioning
    cvPart = cvpartition( Y, 'Holdout', 0.5 );
    XTrn = X( :, training(cvPart) );
    XTst = X( :, test(cvPart)  );
    YTrn = Y( training(cvPart) );
    YTst = Y( test(cvPart)  );

    disp('Generated and partitioned data.');

    % plot the data
    fig1 = figure(1);
    plot( XFd );
    drawnow;

    % plot the first four components
    pcaXFd = pca_fd( XFd, 4 );
    fig2 = figure(2);
    plot_pca_fd( pcaXFd, 1, 1:4 );
    drawnow;


    % ----- autoencoder -----

    % train the autoencoder
    disp('Training autoencoder ... ');
    ae = trainAutoencoder( XTrn, nCodes, ...
                           'MaxEpochs', 1000, ...
                           'ShowProgressWindow', false );

    % generate predictions and calculate errors
    XTrnHat = predict( ae, XTrn );
    XTstHat = predict( ae, XTst );

    errTrn = sqrt( mse( XTrn, XTrnHat ) );
    disp( ['AE Training Error = ' num2str(errTrn)] );
    errTst = sqrt( mse( XTst, XTstHat ) );
    disp( ['AE Testing Error  = ' num2str(errTst)] );

    % plot resulting curves
    XTstHatFd = smooth_basis( setup.data.tFine, XTstHat, setup.fda.fdPar );

    fig3 = figure(3);
    ax = subplot( 2,2,1 );
    subplotFd( ax, XTstHatFd );
    title( ax, 'AE Prediction' );

    % obtain encodings
    ZTrn = encode( ae, XTrn );
    ZTst = encode( ae, XTst );
    ax = subplot( 2,2,3 );
    ZTstCan = cda( ZTst', YTst );

    % classify
    model = fitcdiscr( ZTrn', YTrn );
    YHatTst = predict( model, ZTst' );
    errAE(i) = loss( model, ZTst', YTst );
    disp( ['FITCDISCR: Holdout Loss = ' num2str(errAE(i)) ]);

    % plot the clusters
    plotClusters( ax, ZTstCan.scores, YTst, YHatTst );
    title( ax, 'AE Encoding' );
    drawnow;


    % ----- PCA encoding -----

    disp('Running PCA ... ');
    XTrnFd = smooth_basis( setup.data.tFine, XTrn, setup.fda.fdPar );
    XTstFd = smooth_basis( setup.data.tFine, XTst, setup.fda.fdPar );
    pcaXTrnFd = pca_fd( XTrnFd, nCodes );

    % generate predictions and calculate errors
    ZTrnPCA = pcaXTrnFd.harmscr;
    ZTstPCA = pca_fd_score( XTstFd, ...
                          pcaXTrnFd.meanfd, ...
                          pcaXTrnFd.harmfd, ...
                          nCodes, true );
    
    XTrnFdPCA = pcaXTrnFd.meanfd + pcaXTrnFd.fdhatfd;
    XTstFdPCA = reconstructFd( pcaXTrnFd, ZTstPCA, setup.fda );

    XTrnHatPCA = eval_fd( setup.data.tFine, XTrnFdPCA );
    XTstHatPCA = eval_fd( setup.data.tFine, XTstFdPCA );
    
    errTrnPCA = sqrt( mse( XTrn, XTrnHatPCA ) );
    disp( ['PCA Training Error = ' num2str(errTrnPCA)] );
    errTstPCA = sqrt( mse( XTst, XTstHatPCA ) );
    disp( ['PCA Testing Error  = ' num2str(errTstPCA)] );

    ax = subplot( 2,2,2 );
    subplotFd( ax, XTstFdPCA );
    title( ax, 'PCA Prediction');

    % classify
    modelPCA = fitcdiscr( ZTrnPCA, YTrn );
    YHatTstPCA = predict( modelPCA, ZTstPCA );
    errPCA(i) = loss( modelPCA, ZTstPCA, YTst );
    disp( ['FITCDISCR: Holdout Loss = ' num2str(errPCA(i)) ]);

    % plot the clusters
    ax = subplot( 2,2,4 );
    plotClusters( ax, ZTstPCA, YTst, YHatTstPCA );
    title( ax, 'PCA Encoding' );
    drawnow;

end

disp( ['Mean AE Classification Error  = ' num2str(mean(errAE)) ] );
disp( ['Mean PCA Classification Error = ' num2str(mean(errPCA)) ] );
disp( ['Frequency AE error is lower than PCA error = ' ...
            num2str( sum(errAE<errPCA) ) ] );





