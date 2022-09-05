function XSFd = splitFd( XFd, indices )

    coeff = getcoef( XFd );
    coeff( :, ~indices, : ) = [];
    XSFd = putcoef( XFd, coeff );

end