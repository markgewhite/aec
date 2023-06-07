function dlXCHat = calcPDP( self, dlXC )
    % Generate a Partial Dependence Plot style latent component
    arguments
        self                AEModel
        dlXC                dlarray
    end
   
    XDim = size( dlXC, 1 );
    K = size( dlXC, 2)/self.ZDimAux;
    
    dlXCHat = reshape( dlXC, XDim, K, self.ZDimAux, [] );

end
