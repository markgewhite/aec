% ************************************************************************
% Script: test random convolutional kernels
%
% ************************************************************************

clear;

rng( 0 );
nCodes = 4;
nPts = 21;
nRuns = 10;
dataSource = 'JumpVGRF';

% prepare data
[X, XFd, Y, setup.data ] = initializeData( dataSource, nCodes, nPts ); 
setup.data.isInterdependent = false;
setup.data.nKernels = 1000;
setup.data.smooth = false;

lossTrn = zeros( nRuns, 1 );
lossTst = zeros( nRuns, 1 );
for i = 1:nRuns
    % define kernels
    kernels = generateKernels( size( X,1 ), setup.data );
    
    % apply kernels
    [ XT, conv ] = applyKernels( X, kernels );
    
    % partitioning
    cvPart = cvpartition( Y, 'Holdout', 0.5 );
    XTTrn = XT( :, training(cvPart) )';
    XTTst = XT( :, test(cvPart)  )';
    YTrn = Y( training(cvPart) );
    YTst = Y( test(cvPart)  );
    
    % classify
    mdl = fitclinear( XTTrn( :, select ), YTrn, 'Beta', b );
    lossTrn(i) = loss( mdl, XTTrn( :, select ), YTrn );
    lossTst(i) = loss( mdl, XTTst( :, select ), YTst );

end

disp(['Train Loss = ' num2str( mean(lossTrn), '%.3f' ) ...
              ' +/- ' num2str( std(lossTrn), '%.3f' ) ]);
disp(['Test Loss  = ' num2str( mean(lossTst), '%.3f' ) ...
              ' +/- ' num2str( std(lossTst), '%.3f' ) ]);

% coefficients
kID = 1:2:setup.data.nKernels*2-1;
betaMP = mdl.Beta( kID );
betaPPV = mdl.Beta( kID+1 );

[ betaMP, orderMP ] = sort( abs(betaMP), 'descend' );
lMP = kernels.lengths( orderMP );
dMP = kernels.dilations( orderMP );
cMP = kernels.correlations( orderMP );

[ betaPPV, orderPPV ] = sort( abs(betaPPV), 'descend' );
lPPV = kernels.lengths( orderPPV );
dPPV = kernels.dilations( orderPPV );
cPPV = kernels.correlations( orderPPV );


% beta plots
figure;
ax = subplot(2,2,1);
plot( ax, betaMP );
hold( ax, 'on' );
plot( ax, betaPPV );
hold( ax, 'off' );
legend( ax, {'MP', 'PPV'} );
title( ax, 'Beta' );

% length plots
ax = subplot(2,2,2);
bar( ax, lMP );
hold( ax, 'on' );
bar( ax, lPPV );
hold( ax, 'on' );
legend( ax, {'MP', 'PPV'} );
title( ax, 'Lengths');

% dilation plots
ax = subplot(2,2,3);
plot( ax, dMP );
hold( ax, 'on' );
plot( ax, dPPV );
hold( ax, 'on' );
legend( ax, {'MP', 'PPV'} );
title( ax, 'Dilations');

% correlation plots
ax = subplot(2,2,4);
plot( ax, cMP );
hold( ax, 'on' );
plot( ax, cPPV );
hold( ax, 'on' );
legend( ax, {'MP', 'PPV'} );
title( ax, 'Correlations');






