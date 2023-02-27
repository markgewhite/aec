function XC = getLatentResponse( self, thisDataset )
    % Generate the latent components, smoothing them
    arguments
        self            RepresentationModel            
        thisDataset     ModelDataset
    end

    Z = self.encode( thisDataset, convert = false );
    XC = self.calcLatentComponents( Z ); 

end