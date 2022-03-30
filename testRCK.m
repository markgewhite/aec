% ************************************************************************
% Script: test random convolutional kernels
%
% ************************************************************************

clear;

nCodes = 4;
nPts = 501; % 101 with padding to 1500 ms
doPadding = false;
nRuns = 10;
nFolds = 10;
lambdaRange = logspace( -8, -2, 25 );
dataSource = 'JumpACC';
outcome = 'JumpHeight';

nKernels = 2000;
nMetrics = 4;
sampleRatio = 0.05;

% prepare data
[X, ~, ~, Y, setup.data ] = initializeData( dataSource, nCodes, ...
                                            nPts, outcome ); 
nObs = length( Y );

lossTrn = zeros( nRuns, nFolds );
lossVal = zeros( nRuns, nFolds );
for i = 1:nRuns

    rng( 'shuffle' );
    parameters = fitKernels( X, nKernels, nMetrics, sampleRatio );   
    [ XT, nTrunc ] = rocketTransform( X, parameters );
    
    if i==1
        % initialise beta now knowing how many coefficients
        nFeatures = size( XT, 1 );
        if nTrunc > 0
            disp(['Dilations truncated = ' ...
          num2str(100*nTrunc/(nObs*length(parameters.dilations)), '%.1f') '%']);
        end
    end

    rng( 'default' ); % to generate fixed partitions
    fprintf('Best = ');
    for j = 1:nFolds
    
        % partitioning
        cvPart = cvpartition( length(Y), 'Holdout', 0.5 );
        XTTrn = XT( :, training(cvPart) )';
        XTVal = XT( :, test(cvPart)  )';
        YTrn = Y( training(cvPart) );
        YVal = Y( test(cvPart)  );
        
        switch outcome
            case 'JumpType'
                % classify
                mdl = fitclinear( XTTrn, YTrn, 'Lambda', lambdaRange );
            case {'JumpHeight', 'PeakPower'}
                % regression
                mdl = fitrlinear( XTTrn, YTrn, 'Lambda', lambdaRange );
        end

        % find the best lambda for the training set
        % using the highest lambda if there is a tie
        lossTrnLambda = loss( mdl, XTTrn, YTrn );
        [ lossSorted, orderIdx ] = sort( lossTrnLambda );
        isTie = diff(lossSorted)==0;
        best = orderIdx( find( ~isTie, 1 ) );
        
        lossTrn( i, j ) = lossTrnLambda( best );
        fprintf( ' %2d', best );

        % use training's best lambda for the validation set
        lossValLambda = loss( mdl, XTVal, YVal );
        lossVal( i, j ) = lossValLambda( best );

        beta( i, j, : ) = mdl.Beta( :, best );

    end

    if any(strcmpi( outcome, {'JumpHeight', 'PeakPower'} ))
        % convert to RMSE
        lossTrn(i,:) = sqrt( lossTrn(i,:) );
        lossVal(i,:) = sqrt( lossVal(i,:) );
    end

    fprintf('\n');

    disp(['Train Loss       = ' num2str( mean(lossTrn(i,:)), '%.3f' ) ...
              ' +/- ' num2str( std(lossTrn(i,:)), '%.3f' ) ]);
    disp(['Validation Loss  = ' num2str( mean(lossVal(i,:)), '%.3f' ) ...
              ' +/- ' num2str( std(lossVal(i,:)), '%.3f' ) ]);

end

disp(['Between-Transform Train Loss SD = ' ...
            num2str( std( mean(lossTrn,2) ), '%.3f' ) ]);
disp(['Between-Transform Test Loss SD  = ' ...
            num2str( std( mean(lossVal,2) ), '%.3f' ) ]);
disp(['Between-Fold Train Loss SD = ' ...
            num2str( std( mean(lossTrn,1) ), '%.3f' ) ]);
disp(['Between-Fold Test Loss SD  = ' ...
            num2str( std( mean(lossVal,1) ), '%.3f' ) ]);

disp(['Overall Train Loss = ' num2str( mean(lossTrn,'all'), '%.3f' ) ...
              ' +/- ' num2str( std(lossTrn,[],'all'), '%.3f' ) ]);
disp(['Overall Test Loss  = ' num2str( mean(lossVal,'all'), '%.3f' ) ...
              ' +/- ' num2str( std(lossVal,[],'all'), '%.3f' ) ]);

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











