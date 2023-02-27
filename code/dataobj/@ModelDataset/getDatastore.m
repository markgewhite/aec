function dsFull = getDatastore( self, includeP, getCoeff )
    % Create a super datastore of individual datastores for each variable
    arguments
        self            ModelDataset
        includeP        logical = false
        getCoeff        logical = false
    end  

    if getCoeff
        X = self.XInputCoeff;
        XN = self.XTargetCoeff;
    else
        X = self.XInput;
        XN = self.XTarget;
    end
    
    if includeP
        P = calcXDistribution( self.XTarget, self.Perplexity );
    else
        P = zeros( self.NumObs, self.NumObs );
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