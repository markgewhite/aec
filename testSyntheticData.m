% ************************************************************************
% Script: test synthetic data
%
% ************************************************************************

clear;

setup.fda.basisOrder = 4;
setup.fda.penaltyOrder = 2;
setup.fda.lambda = 1E2;
setup.fda.nBasis = 50+setup.fda.penaltyOrder+1;

classSizes = [ 100 100 100 ];
nDim = 1;
setup.data.tFine = linspace( 0, 1000, 100 );
setup.data.tSpan = linspace( 0, 1024, 33 );
setup.data.ratio = [ 4 8 16];
setup.data.sharedLevel = 3;
setup.data.mu = [1 4 8];
setup.data.sigma = [1 1 1];
setup.data.eta = 0.1;
setup.data.warpLevel = 2;
setup.data.tau = 20;

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
plot_pca_fd( pcaXFd );

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
