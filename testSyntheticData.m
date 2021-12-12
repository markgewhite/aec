% ************************************************************************
% Script: test synthetic data
%
% ************************************************************************

setup.nBasis = 43;
setup.basisOrder = 4;
setup.penaltyOrder = 2;
setup.lambda = 10E0;
setup.constraints = [ 0, 1; 1000, 0 ];

tSpan = linspace( 0, 1000, 100 );

basis = create_bspline_basis( [ tSpan(1), tSpan(end) ], ...
                              setup.nBasis, ...
                              setup.basisOrder);
                          
XFdPar = fdPar( basis, setup.penaltyOrder, setup.lambda ); 

basisHL = create_bspline_basis( [ tSpan(1), tSpan(end) ], ...
                              (setup.nBasis-3)/4+3, ...
                              setup.basisOrder);

basisHL2 = create_bspline_basis( [ tSpan(1), tSpan(end) ], ...
                              (setup.nBasis-3)/8+3, ...
                              setup.basisOrder);

[X, XFd] = genSyntheticData( [ 20 20 20 ], 3, ...
                                { basis, basisHL, basisHL2 }, ...
                                [10 2 5], ...
                                setup.constraints );

plot( XFd );