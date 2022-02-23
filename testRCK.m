% ************************************************************************
% Script: test random convolutional kernels
%
% ************************************************************************

clear;

rng( 0 );
nCodes = 4;
nPts = 101;
nRuns = 10;
nFolds = 10;
dataSource = 'JumpVGRF';
method = 'PPV';
algorithm = 'MiniRocket';

% prepare data
[X, XFd, Y, setup.data ] = initializeData( dataSource, nCodes, nPts ); 
setup.data.isInterdependent = false;
setup.data.nKernels = 2000;
setup.data.nMetrics = 3;
setup.data.smooth = true;

lossTrn = zeros( nRuns, nFolds );
lossTst = zeros( nRuns, nFolds );
for i = 1:nRuns

    switch algorithm
        case 'Rocket'
            kernels = generateKernels( size( X,1 ), setup.data );   
            XT = applyKernels( X, kernels, method );
            setup.data.nMetrics = size( XT, 1 )/setup.data.nKernels;

        case 'MiniRocket'
            kernels = fitKernels( X, setup.data.nKernels, setup.data.nMetrics );   
            XT = rocketTransform( X, kernels );
    end


    for j = 1:nFolds
    
        % partitioning
        cvPart = cvpartition( Y, 'Holdout', 0.5 );
        XTTrn = XT( :, training(cvPart) )';
        XTTst = XT( :, test(cvPart)  )';
        YTrn = Y( training(cvPart) );
        YTst = Y( test(cvPart)  );
        
        % classify
        mdl = fitclinear( XTTrn, YTrn, ...
                          'Lambda', logspace(-8,0,33) );
        [ ~, best ] = min(loss( mdl, XTTst, YTst ));
        [~, orderID] = sort( mdl.Beta(:,best) );
        select = orderID <= size( XTTrn, 2 )*0.8;
        mdl = fitclinear( XTTrn( :, select ), YTrn, ...
                          'Lambda', logspace(-8,0,33) );

        [ lossTst( i, j ), best ] = min(loss( mdl, XTTst( :, select), YTst ));
        disp(['Best = ' num2str(best)]);
        lossTrnLambda = loss( mdl, XTTrn( :, select), YTrn );
        lossTrn( i, j ) = lossTrnLambda( best );

    end
    
    disp(['Train Loss = ' num2str( mean(lossTrn(i,:)), '%.3f' ) ...
              ' +/- ' num2str( std(lossTrn(i,:)), '%.3f' ) ]);
    disp(['Test Loss  = ' num2str( mean(lossTst(i,:)), '%.3f' ) ...
              ' +/- ' num2str( std(lossTst(i,:)), '%.3f' ) ]);

end

disp(['Between-Transform Train Loss SD = ' ...
            num2str( std( mean(lossTrn,2) ), '%.3f' ) ]);
disp(['Between-Transform Test Loss SD  = ' ...
            num2str( std( mean(lossTst,2) ), '%.3f' ) ]);
disp(['Between-Fold Train Loss SD = ' ...
            num2str( std( mean(lossTrn,1) ), '%.3f' ) ]);
disp(['Between-Fold Test Loss SD  = ' ...
            num2str( std( mean(lossTst,1) ), '%.3f' ) ]);

disp(['Overall Train Loss = ' num2str( mean(lossTrn,'all'), '%.3f' ) ...
              ' +/- ' num2str( std(lossTrn,[],'all'), '%.3f' ) ]);
disp(['Overall Test Loss  = ' num2str( mean(lossTst,'all'), '%.3f' ) ...
              ' +/- ' num2str( std(lossTst,[],'all'), '%.3f' ) ]);

% coefficients
kID = 1 : setup.data.nMetrics : setup.data.nKernels*setup.data.nMetrics-1;
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






