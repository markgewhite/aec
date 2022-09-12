function [ XA, Q, XC ] = getLatentResponse( self, thisDataset )
    % Generate the latent components and
    % compute the explained variance
    arguments
        self            RepresentationModel            
        thisDataset     ModelDataset
    end

    % calculate the auxiliary model/network dependence
    [XA, Q] = self.getAuxALE( thisDataset );

    % generate the components, smoothing them, for storage
    Z = self.encode( thisDataset, convert = false );
    XC = self.calcLatentComponents( Z, smooth = true ); 

end