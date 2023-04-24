function XHat = constructCurves( tSpan, meanFd, compFd, Z )
    % Construct the curves from the mean and components
    % by linearly summing the components, points-wise
    arguments
        tSpan       double
        meanFd
        compFd
        Z           double
    end

    % create the set of points from the mean for each curve
    nRows = size( Z, 1 );
    XHat = repmat( eval_fd( tSpan, meanFd ), 1, nRows );

    % generate components as sets of points
    XC = eval_fd( tSpan, compFd );  

    % linearly combine the components, points-wise
    nChannels = size( XC, 3 );
    nDim = size( XC, 2 );

    for k = 1:nChannels
        for j = 1:nDim        
            for i = 1:nRows
                XHat(:,i,k) = XHat(:,i,k) + Z(i,(k-1)*nDim+j)*XC(:,j,k);
            end
        end
    end

end