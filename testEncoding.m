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

adversarialDesign = true;

rng( 'default' );

% data generation parameters
setup.data.tFine = linspace( 0, 1000, 21 );
setup.data.tSpan = linspace( 0, 1024, 33 );
setup.data.ratio = [ 4 8 16];
setup.data.sharedLevel = 3;
setup.data.mu = [1 4 8];
setup.data.sigma = [1 6 1];
setup.data.eta = 0.1;
setup.data.warpLevel = 2;
setup.data.tau = 0;

% functional data analysis parameters
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

% AAE training parameters
setup.aae.designFcn = @aaeDesign;
setup.aae.gradFcn = @modelGradients;
setup.aae.nEpochs = 1000; 
setup.aae.batchSize = 20;
setup.aae.beta1 = 0.9;
setup.aae.beta2 = 0.999;
setup.aae.L2Regularization = 0.001;
setup.aae.valFreq = 100;
setup.aae.valSize = [2 5];
setup.aae.distSize = 500;
setup.aae.zDim = nCodes;
setup.aae.xDim = length( setup.data.tFine );

% encoder network parameters
setup.aae.enc.learnRate = 0.005;
setup.aae.enc.scale = 0.2;
setup.aae.enc.input = setup.aae.xDim;
setup.aae.enc.outZ = setup.aae.zDim;

% decoder network parameters
setup.aae.dec.learnRate = 0.005;
setup.aae.dec.scale = 0.2;
setup.aae.dec.input = setup.aae.zDim;
setup.aae.dec.outX = setup.aae.xDim;


% initialise plots
figure(3);
ax.ae.pred = subplot( 2,2,1 );
ax.pca.pred = subplot( 2,2,2 );
ax.ae.cls = subplot( 2,2,3 );
ax.pca.cls = subplot( 2,2,4 );
figure(4);
ax.ae.distZTrn = subplot( 2,2,1 );
ax.pca.distZTrn = subplot( 2,2,2 );
ax.ae.distZTst = subplot( 2,2,3 );
ax.pca.distZTst = subplot( 2,2,4 );
figure(5);
ax.ae.comp = gobjects( nCodes, 1 );
[ rows, cols ] = sqdim( nCodes );
for i = 1:nCodes
    ax.ae.comp(i) = subplot( rows, cols, i );
end

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
    if adversarialDesign
        [dlnetEnc, dlnetDec] = trainAAE( XTrn, setup.aae );
    else
        disp('Training autoencoder ... ');
        ae = trainAutoencoder( XTrn, nCodes, ...
                               'MaxEpochs', 1000, ...
                               'ShowProgressWindow', false );
    end

    % switch to DL array format
    dlXTrn = dlarray( XTrn, 'CB' );
    dlXTst = dlarray( XTst, 'CB' );

    % generate encodings
    dlZTrn = predict( dlnetEnc, dlXTrn );
    dlZTst = predict( dlnetEnc, dlXTst );

    % convert back to numeric arrays
    ZTrn = double(extractdata( dlZTrn ));
    ZTst = double(extractdata( dlZTst ));

    % plot Z distribution 
    plotZDist( ax.ae.distZTrn, ZTrn, 'AE: Z Train', true );
    plotZDist( ax.ae.distZTst, ZTst, 'AE: Z Test', true );

    % plot characteristic features
    plotLatentComp( ax.ae.comp, dlnetDec, ZTrn, ...
                    setup.data.tFine, setup.fda.fdPar );

    % classify
    model = fitcdiscr( ZTrn', YTrn );
    YHatTst = predict( model, ZTst' );
    errAE(i) = loss( model, ZTst', YTst );
    disp( ['FITCDISCR: Holdout Loss = ' num2str(errAE(i)) ]);

    % plot the clusters for test data using the training loadings 
    % from a canonical discriminant analysis
    ZTrnNse = ZTrn + 1E-6*randn(size(ZTrn)); % to avoid errors
    ZTrnCanInfo = cda( ZTrn', YTrn );
    ZTstCan = ZTst'*ZTrnCanInfo.loadings;
    plotClusters( ax.ae.cls, ZTstCan, YTst, YHatTst );
    title( ax.ae.cls, 'AE Encoding' );
    drawnow;

    % reconstruct the cures and calculate errors
    dlXTrnHat = predict( dlnetDec, dlZTrn );
    dlXTstHat = predict( dlnetDec, dlZTst );
    XTrnHat = double(extractdata( dlXTrnHat ));
    XTstHat = double(extractdata( dlXTstHat ));

    errTrn = sqrt( mse( XTrn, XTrnHat ) );
    disp( ['AE Training Error = ' num2str(errTrn)] );
    errTst = sqrt( mse( XTst, XTstHat ) );
    disp( ['AE Testing Error  = ' num2str(errTst)] );

    % plot resulting curves
    XTstHatFd = smooth_basis( setup.data.tFine, XTstHat, setup.fda.fdPar );

    subplotFd( ax.ae.pred, XTstHatFd );
    title( ax.ae.pred, 'AE Prediction' );



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

    subplotFd( ax.pca.pred, XTstFdPCA );
    title( ax.pca.pred, 'PCA Prediction');

    % plot Z distribution 
    plotZDist( ax.pca.distZTrn, ZTrnPCA', 'PCA: Z Train', true );
    plotZDist( ax.pca.distZTst, ZTstPCA', 'PCA: Z Test', true );

    % classify
    modelPCA = fitcdiscr( ZTrnPCA, YTrn );
    YHatTstPCA = predict( modelPCA, ZTstPCA );
    errPCA(i) = loss( modelPCA, ZTstPCA, YTst );
    disp( ['FITCDISCR: Holdout Loss = ' num2str(errPCA(i)) ]);

    % plot the clusters
    plotClusters( ax.pca.cls, ZTstPCA, YTst, YHatTstPCA );
    title( ax.pca.cls, 'PCA Encoding' );
    drawnow;

    pause;

end

disp( ['Mean AE Classification Error  = ' num2str(mean(errAE)) ] );
disp( ['Mean PCA Classification Error = ' num2str(mean(errPCA)) ] );
disp( ['Frequency AE error is lower than PCA error = ' ...
            num2str( sum(errAE<errPCA) ) ] );





