% ************************************************************************
% Script: test synthetic data
%
% ************************************************************************

clear;

action = 'Autoencoder';

setup.fda.basisOrder = 4;
setup.fda.penaltyOrder = 2;
setup.fda.lambda = 1E2;
setup.fda.nBasis = 20+setup.fda.penaltyOrder+1;

N = 100;
classSizes = [ N N N ];
nDim = 1;
nCodes = 10;
setup.data.tFine = linspace( 0, 1000, 101 );
setup.data.tSpan = linspace( 0, 1024, 33 );
setup.data.ratio = [ 4 8 16];
setup.data.sharedLevel = 3;
setup.data.mu = [1 4 8];
setup.data.sigma = [1 6 1];
setup.data.eta = 0.1;
setup.data.warpLevel = 2;
setup.data.tau = 0;

setup.reg.nIterations = 2;
setup.reg.nBasis = 12; 
setup.reg.basisOrder = 3; 
setup.reg.wLambda = 1E-2; 
setup.reg.XLambda = 1E2;
setup.reg.usePC = true;
setup.reg.nPC = 3;


rng( 'default' );
Xraw = genSyntheticData( classSizes, ...
                      nDim, ...
                      setup.data );

setup.basisFd = create_bspline_basis( ...
                        [ setup.data.tSpan(1), setup.data.tSpan(end) ], ...
                          setup.fda.nBasis, setup.fda.basisOrder);

XFdPar = fdPar( setup.basisFd, setup.fda.penaltyOrder, setup.fda.lambda ); 

XFd = smooth_basis( setup.data.tSpan, Xraw, XFdPar );

figure(1);
clf;
plot( XFd );

X = eval_fd( setup.data.tFine, XFd );

pcaXFd = pca_fd( XFd, 10 );
figure(2);
clf;
plot_pca_fd( pcaXFd, 1, 1:4 );

%load('Postdoc/sandbox/regData', ...
%      'XFdReg', 'warpFd' );
%pcaXFd = pca_fd( XFdReg, 10 );
%pcaWFd = pca_fd( warpFd, 5 );

Y = [ repelem(1,N) repelem(2,N) repelem(3,N) ]';
%Z = [ pcaXFd.harmscr pcaWFd.harmscr ];
Z = pcaXFd.harmscr;


switch action
    
    case 'Registration'
        
        setup.reg.usePC = true;
        [ XFdReg, warpFd ] = registerCurves( setup.data.tSpan, XFd, ...
                                                    'Continuous', setup.reg, [] );
        
        figure(3);
        clf;
        plot( XFdReg );
        drawnow;
        
        setup.reg.usePC = false;
        [ XFdRegNonPC, warpFdNonPC ] = registerCurves( setup.data.tSpan, XFd, ...
                                                    'Continuous', setup.reg, [] );
        
        figure(4);
        clf;
        plot( XFdRegNonPC );

        save('Postdoc/sandbox/regData', ...
                 'XFd', 'XFdReg', 'warpFd', ...
                 'XFdRegNonPC', 'warpFdNonPC', ...
                 'pcaXFd', ...
                 'setup' );


    case 'Classification'

        model = fitcdiscr( Z, Y, 'CrossVal', 'on' );

        Yhat = kfoldPredict( model );
        error = kfoldLoss( model );
        disp( ['FITCDISCR: KFold Loss = ' num2str(error) ]);

        % model = fitcauto( Z, Y );

    case 'Autoencoder'

        % partitioning
        cvPart = cvpartition( Y, 'Holdout', 0.5 );
        XTrn = X( :, training(cvPart) );
        XTst = X( :, test(cvPart)  );

        % train the autoencoder
        ae = trainAutoencoder( XTrn, nCodes, ...
                               'MaxEpochs', 4000 );

        % generate predictions and calculate errors
        XTrnHat = predict( ae, XTrn );
        XTstHat = predict( ae, XTst );

        errTrn = sqrt( mse( XTrn, XTrnHat ) );
        disp( ['AE Training Error = ' num2str(errTrn)] );
        errTst = sqrt( mse( XTst, XTstHat ) );
        disp( ['AE Testing Error  = ' num2str(errTst)] );

        % plot resulting curves
        XTstHatFd = smooth_basis( setup.data.tFine, XTstHat, XFdPar );

        figure(3);
        clf;
        plot( XTstHatFd );
        title( 'AE Prediction');


        % PCA encoding

        XTrnFd = smooth_basis( setup.data.tFine, XTrn, XFdPar );
        XTstFd = smooth_basis( setup.data.tFine, XTst, XFdPar );
        pcaXTrnFd = pca_fd( XTrnFd, nCodes );

        % generate predictions and calculate errors
        ZTrn = pcaXTrnFd.harmfd;
        ZTst = pca_fd_score( XTstFd, ...
                              pcaXTrnFd.meanfd, ...
                              pcaXTrnFd.harmfd, ...
                              nCodes, true );
        
        XTrnFdPCA = pcaXTrnFd.meanfd + pcaXTrnFd.fdhatfd;
        XTstFdPCA = reconstructFd( pcaXTrnFd, ZTst, setup.fda );

        XTrnHatPCA = eval_fd( setup.data.tFine, XTrnFdPCA );
        XTstHatPCA = eval_fd( setup.data.tFine, XTstFdPCA );
        
        errTrnPCA = sqrt( mse( XTrn, XTrnHatPCA ) );
        disp( ['PCA Training Error = ' num2str(errTrnPCA)] );
        errTstPCA = sqrt( mse( XTst, XTstHatPCA ) );
        disp( ['PCA Testing Error  = ' num2str(errTstPCA)] );

        figure(4);
        clf;
        plot( XTstFdPCA );
        title( 'PCA Prediction');

        XTstErrFdPCA = smooth_basis( setup.data.tFine, ...
                                        XTstHatPCA-XTst, XFdPar );


    otherwise

        disp('Action not recognised.');

end




