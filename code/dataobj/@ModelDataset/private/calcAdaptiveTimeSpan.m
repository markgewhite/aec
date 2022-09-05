function tSpanAdaptive = calcAdaptiveTimeSpan( XFd, tSpan, ...
                                               lowerBound, upperBound ) 

    % evaluate the mean XFd curvature (2nd derivative)
    D1XEval = squeeze(mean( abs(eval_fd( tSpan, XFd, 1 )), 2));
    D2XEval = squeeze(mean( abs(eval_fd( tSpan, XFd, 2 )), 2));

    D1XEval = min( max( D1XEval./mean(D1XEval), lowerBound ), upperBound );
    D2XEval = min( max( D2XEval./mean(D2XEval), lowerBound ), upperBound );
    
    DXEvalComb = sum( D1XEval + D2XEval, 2 );

    % cumulatively sum the absolute inverse curvatures
    % inserting zero at the begining to ensure first point will be at 0
    D2XInt = cumsum( [0; 1./DXEvalComb] );
    D2XInt = D2XInt./max(D2XInt);

    % normalize to the tSpan
    tSpanAdaptive = tSpan(1) + D2XInt*(tSpan(end)-tSpan(1));

    % reinterpolate to remove the extra point
    nPts = length( tSpan );
    tSpanAdaptive = interp1( 1:nPts+1, ...
                             tSpanAdaptive, ...
                             linspace(1, nPts+1, nPts) );

end