% ************************************************************************
% Script: test random convolutional kernels
%
% ************************************************************************

clear;

nCodes = 4;
nPts = 1+512/4; % 101 with padding to 1500 ms
doPadding = false;
nRuns = 10;
nFolds = 10;
dataSource = 'JumpVGRF';
method = 'PPV';
algorithm = 'MiniRocket';
nKernels = 2000;
nMetrics = 4;
sampleRatio = 0.05;
useSubset = false;
subsetProp = 0.8;

% prepare data
[X, XFd, Y, setup.data ] = initializeData( dataSource, nCodes, ...
                                            nPts, doPadding ); 
nObs = length( Y );

lossTrn = zeros( nRuns, nFolds );
lossTst = zeros( nRuns, nFolds );
for i = 1:nRuns

    rng( 'shuffle' );
    parameters = fitKernels( X, nKernels, nMetrics, sampleRatio );   
    [ XT, nTrunc ] = rocketTransform( X, parameters );
    
    if i==1
        % initialise beta now knowing how many coefficients
        nFeatures = size( XT, 1 );
        if useSubset
            beta = zeros( nRuns, nFolds, fix(nFeatures*subsetProp) );
        else
            beta = zeros( nRuns, nFolds, nFeatures );
        end
        if nTrunc > 0
        disp(['Dilations truncated = ' ...
          num2str(100*nTrunc/(nObs*length(parameters.dilations)), '%.1f') '%']);
        end
    end

    rng( 'default' ); % to generate fixed partitions
    fprintf('Best = ');
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

        if useSubset
            % select highest ranked features
            [ ~, best ] = min(loss( mdl, XTTst, YTst ));
            [~, orderID] = sort( mdl.Beta(:,best) );
            selection = orderID <= nFeatures*subsetProp;
            mdl = fitclinear( XTTrn( :, selection ), YTrn, ...
                              'Lambda', logspace(-8,0,33) );
        else
            % use all features
            selection = true( nFeatures, 1 );
        end

        [ lossTst( i, j ), best ] = min(loss( mdl, XTTst( :, selection), YTst ));
        fprintf( ' %2d', best );
        lossTrnLambda = loss( mdl, XTTrn( :, selection), YTrn );
        lossTrn( i, j ) = lossTrnLambda( best );

        beta( i, j, : ) = mdl.Beta( :, best );

    end
    fprintf('\n');

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

% beta plots
figure;
palette = lines(7);
metrics = {'PPV', 'MPV', 'MIPV', 'LSPV' };

kID = 0 : nMetrics : nFeatures-nMetrics-1;
for m = 1:nMetrics
    kID = kID + 1;
    colour = palette(m,:);
    ax = subplot(2,2,m);
    hold( ax, 'on' );
    betaSorted = zeros( length(kID), 1 );
    for i = 1:nRuns
        for j = 1:nFolds
            betaS = sort( abs(beta( i, j, kID )), 'descend' );
            betaS = squeeze(  betaS );
            plot( ax, betaS, 'Color', colour );
            betaSorted = betaSorted + betaS;
        end
    end
    plot( ax, betaSorted/(nRuns*nFolds), 'Color', [0 0 0], 'LineWidth', 2 );
    hold( ax, 'off' );
    ax.YScale = 'log';
    grid( ax, 'on' );
    ylim( ax, [0.001 1] );
    title( ax, metrics(m) );
end











