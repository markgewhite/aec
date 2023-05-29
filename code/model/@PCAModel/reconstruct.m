function [ XHat, XHatSmth ] = reconstruct( self, Z, args )
    % Reconstruct X from Z using the model
    arguments
        self                PCAModel
        Z                   double  % latent codes
        args.smooth         logical % redundant
    end
           
    XHat = constructCurves( self.TSpan.Target, ...
                            self.MeanFd, self.CompFd, ...
                            Z );
    
    XHatSmth = XHat;

end