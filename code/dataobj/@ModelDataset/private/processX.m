function [ X, XDim ] = processX( ...
                                XFd, XLen, tSpan, tSpanNew, pad, ...
                                normalize, normalizedPts, normalization )

    % evaluate the input function at these points
    XEval = eval_fd( tSpanNew, XFd );
   
    % adjust lengths for non-linear re-sampling
    XLenNew = adjustXLengths( XLen, tSpan, tSpanNew, pad.Location );
    
    % re-scale for resampled length
    pad.Length = max( XLenNew );
    
    % recreate the cell time series
    XCell = extractXSeries( XEval, XLenNew, pad.Length, pad.Location );

    if normalize
        % use time-normalization method to set a fixed length
        XNorm = normalizeXSeries( XCell, normalizedPts, ...
                                        normalization, ...
                                        pad );
        XDim = size( XNorm, 1);
        if size( XNorm, 3 ) > 1
            X = num2cell( permute( XNorm, [2 1 3]), [2 3] );
            X = cellfun( @squeeze, X , 'UniformOutput', false);
        else
            X = num2cell( permute( XNorm, [2 1]), 2 );
            X = cellfun( @transpose, X , 'UniformOutput', false);
        end
        
    else
        % has variable length input
        X = XCell;
        XDim = 1;
    end

end