function validateSmoothing( thisModel, args )
    % Re-run smoothing with maximum flexibility
    arguments
        thisModel       ModelDataset
        args.X          double = []
        args.TSpan      double = []
    end
    
    if isempty( args.X )
        % default: use the raw input series
        % pad the raw series for smoothing
        X = padData( thisModel.XInputRaw, ...
                     thisModel.Padding.Length, ...
                     thisModel.Padding.Value, ...
                     Same = thisModel.Padding.Same, ...
                     Location = thisModel.Padding.Location, ...
                     Anchoring = thisModel.Padding.Anchoring );
    else
        % use the specified X
        X = args.X;
    
    end
    
    if isempty( args.TSpan )
        % default: use the orginal time series
        % create a time span with maximum detail
        thisTSpan = linspace( thisModel.TSpan.Original(1),...
                      thisModel.TSpan.Original(end), ...
                      size( X, 1 ) );
    else
        % use the specified timespan
        thisTSpan = args.TSpan;
    end
    
    % create a new basis with maximum number of functions
    basis = create_bspline_basis( [thisTSpan(1) thisTSpan(end)], ...
                                  size( X, 1 ), ...
                                  thisModel.FDA.BasisOrder);
    
    % Find minimum GCV value of lambda
    % search for the best value for lambda, the roughness penalty
    logLambda   = -10:1:10;
    gcvSave = zeros( length(logLambda), thisModel.XChannels );
    dfSave  = zeros( length(logLambda), 1 );
    
    for i = 1:length(logLambda)
        
        % set smoothing parameters
        XFdPari = fdPar( basis, ...
                         thisModel.FDA.PenaltyOrder, ...
                         10^logLambda(i) );
        
        % perform smoothing
        [~, dfi, gcvi] = smooth_basis( thisTSpan, X, XFdPari );
        
        % determine mean GCV and degrees of freedom
        gcvSave(i,:) = sqrt( sum( gcvi, 2 )/thisModel.NumObs ); 
        dfSave(i)  = dfi;
        
    end
    
    %  plot the results for GCV and DF
    figure;
    
    plot( logLambda, log10(gcvSave), 'k-o' );
    ylabel('\fontsize{13} log_{10}( GCV )');
    hold on;
    
    yyaxis right;
    plot( logLambda, log10(dfSave), 'r-o' );
    ylabel('\fontsize{13} log_{10}( DF )');
    
    xlabel('\fontsize{13} log_{10}(\lambda)');

end