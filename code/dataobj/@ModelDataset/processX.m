function [ X, XDim ] = processX( self, tSpanNew, normalize, normalizedPts )
    % Process XFd to the required time span
    arguments
        self            ModelDataset
        tSpanNew        double
        normalize       logical
        normalizedPts   double
    end

    % evaluate the input function at these points
    % using single not double to save on memory
    XEval = single(eval_fd( tSpanNew, self.XFd ));
    
    % recreate the cell time series
    XCell = extractXSeries( XEval, ...
                            self.XLen, ...
                            self.Padding.Length, ...
                            self.Padding.Location );

    if normalize
        % use time-normalization method to set a fixed length
        XNorm = normalizeXSeries( XCell, normalizedPts, ...
                                         self.Normalization, ...
                                         self.Padding );
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