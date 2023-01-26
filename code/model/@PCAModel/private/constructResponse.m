function XHat = constructResponse( tSpan, meanFd, compFd, Z )
    % Wrapper response function for constructing curves
    % performing a transpose prior to call
    arguments
        tSpan       double
        meanFd
        compFd
        Z           double
    end

    XHat = constructCurves( tSpan, meanFd, compFd, Z' );

end