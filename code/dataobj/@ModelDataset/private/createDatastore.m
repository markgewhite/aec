function dsFull = createDatastore( X, XN, P, Y, doSort )
   
    if doSort
        % sort them in ascending order of length
        XLen = cellfun( @length, X );
        [ ~, orderIdx ] = sort( XLen, 'descend' );
        X = X( orderIdx );
        XN = XN( orderIdx );
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