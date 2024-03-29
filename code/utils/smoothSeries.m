function XOut = smoothSeries( XIn, tSpanIn, tSpanOut, fdParams )
    % Smooth a time series
    arguments
        XIn         {mustBeA(XIn, {'double', 'single', 'dlarray'})}
        tSpanIn     double
        tSpanOut    double
        fdParams    
    end

    XFd = smooth_basis( tSpanIn, XIn, fdParams );
    XOut = eval_fd( tSpanOut, XFd );

end