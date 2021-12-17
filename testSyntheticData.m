% ************************************************************************
% Script: test synthetic data
%
% ************************************************************************

setup.basisOrder = 4;
setup.penaltyOrder = 2;
setup.lambda = 10E0;
setup.nBasis = 32+setup.penaltyOrder+1;

tSpan = linspace( 0, 1024, 128 );

basis = create_bspline_basis( [ tSpan(1), tSpan(end) ], ...
                              setup.nBasis, ...
                              setup.basisOrder);
                          
XFdPar = fdPar( basis, setup.penaltyOrder, setup.lambda ); 

adjust = setup.penaltyOrder+1;
basisHL = create_bspline_basis( [ tSpan(1), tSpan(end) ], ...
                              (setup.nBasis-adjust)/4+adjust, ...
                              setup.basisOrder);

basisHL2 = create_bspline_basis( [ tSpan(1), tSpan(end) ], ...
                              (setup.nBasis-adjust)/8+adjust, ...
                              setup.basisOrder);

[X, XFd] = genSyntheticData( [ 200 200 200 ], 3, ...
                                { basis, basisHL, basisHL2 }, ...
                                [1 2 4], [0.25 0.0 1], 50 );

plot( XFd );