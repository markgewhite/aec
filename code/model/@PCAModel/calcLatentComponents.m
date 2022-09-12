function [ XC, XMean, offsets ] = calcLatentComponents( self, Z, args )
    % Present the FPCs in form consistent with autoencoder model
    arguments
        self            PCAModel
        Z               double % redundant
        args.forward    logical = false % redundant
        args.smooth     logical = false % redundant
    end

    % compute the components
    nSample = self.NumCompLines;
    offsets = norminv(linspace( 0.050, 0.950, nSample ));
    
    % XC structure: Points, Samples, Components, Channels
    XC = zeros( length(self.PCATSpan), nSample, self.ZDim, self.XChannels );

    for i =1:self.ZDim
        FPC = squeeze(eval_fd( self.PCATSpan, self.CompFd(i) ));
        for c = 1:self.XChannels
            for j = 1:nSample
                XC(:,j,i,c) = offsets(j)*FPC(:,c);
            end
        end
    end

    XMean = squeeze(eval_fd( self.PCATSpan, self.MeanFd ));

end

