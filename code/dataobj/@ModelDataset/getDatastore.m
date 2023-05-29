function dsFull = getDatastore( self, tSpan, includeP )
    % Create a super datastore of individual datastores for each variable
    arguments
        self            ModelDataset
        tSpan           double
        includeP        logical = false
    end  

    X = self.XInputCell;
    XN = self.XTarget( tSpan );
    
    if includeP
        P = calcXDistribution( self.XTarget, self.Perplexity );
    else
        P = single(zeros( self.NumObs, 1 ));
    end
    Y = self.Y;

    if ~self.HasNormalizedInput
        % sort them in ascending order of length
        [ ~, orderIdx ] = sort( self.NumObs, 'descend' );
        X = X( orderIdx );
        XN = XN( orderIdx );
        P = P( orderIdx );
        Y = Y( orderIdx );
    end

    % create the datastore for the input X
    dsX = arrayDatastore( X, 'IterationDimension', 1, ...
                          OutputType = 'same' );
       
    % create the datastore for the time-normalised output X
    dsXN = arrayDatastore( XN, 'IterationDimension', 2 );

    % create the datastore for the P distribution
    dsP = arrayDatastore( P, 'IterationDimension', 1 );
    
    % create the datastore for the labels/outcomes
    dsY = arrayDatastore( Y, 'IterationDimension', 1 );

    % create the datastore for indexing to support P
    dsI = arrayDatastore( 1:length(Y), 'IterationDimension', 2 );
    
    % combine them
    dsFull = combine( dsX, dsXN, dsP, dsY, dsI );
               
end