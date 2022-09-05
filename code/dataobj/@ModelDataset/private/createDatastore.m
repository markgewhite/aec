function dsFull = createDatastore( X, XN, Y )

    % create the datastore for the input X
         
    % sort them in ascending order of length
    %XLen = cellfun( @length, X );
    %[ ~, orderIdx ] = sort( XLen, 'descend' );

    %X = X( orderIdx );
    dsX = arrayDatastore( X, 'IterationDimension', 1, ...
                             'OutputType', 'same' );
       
    % create the datastore for the time-normalised output X
    dsXN = arrayDatastore( XN, 'IterationDimension', 2 );
    
    % create the datastore for the labels/outcomes
    dsY = arrayDatastore( Y, 'IterationDimension', 1 );   
    
    % combine them
    dsFull = combine( dsX, dsXN, dsY );
               
end