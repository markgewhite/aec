% ************************************************************************
% Script: test synthetic data
%
% ************************************************************************

clear;

setup.fda.basisOrder = 4;
setup.fda.penaltyOrder = 2;
setup.fda.lambda = 1E2;
setup.fda.nBasis = 50+setup.fda.penaltyOrder+1;

classSizes = [ 200 200 200 ];
nDim = 3;
setup.data.tFine = linspace( 0, 1000, 100 );
setup.data.tSpan = linspace( 0, 1024, 33 );
setup.data.ratio = [ 1 4 8 16];
setup.data.mu = [1 2 4 8];
setup.data.sigma = [0.25 0.5 0.75 1];
setup.data.eta = 0.1;
setup.data.warpLevel = 3;
setup.data.tau = 100;


Xraw = genSyntheticData( classSizes, ...
                      nDim, ...
                      setup.data );

setup.basisFd = create_bspline_basis( ...
                        [ setup.data.tSpan(1), setup.data.tSpan(end) ], ...
                          setup.fda.nBasis, setup.fda.basisOrder);

XFdPar = fdPar( setup.basisFd, setup.fda.penaltyOrder, setup.fda.lambda ); 

XFd = smooth_basis( setup.data.tSpan, Xraw, XFdPar );

plot( XFd );

X = eval_fd( setup.data.tFine, XFd );