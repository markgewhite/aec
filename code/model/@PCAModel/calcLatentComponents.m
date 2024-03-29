function [ XC, XMean, zs ] = calcLatentComponents( self, Z )
    % Calculate the functional principal components
    % Or use the response function
    arguments
        self                PCAModel
        Z                   double
    end

    % compute the components
    nSample = self.NumCompLines;
    % set z-score levels
    zs = linspace( -2, 2, nSample );

    % calculate the standard deviation
    ZSD = std( Z );
    
    % XC structure: Points, Samples, Components, Channels
    XC = zeros( length(self.TSpan.Target), nSample, self.ZDim, self.XChannels );

    for i =1:self.ZDim
        FPC = squeeze(eval_fd( self.TSpan.Target, self.CompFd(i) ));
        for c = 1:self.XChannels
            for j = 1:nSample
                XC(:,j,i,c) = zs(j)*ZSD(1,(c-1)*self.ZDim+i)*FPC(:,c);
            end
        end
    end

    XMean = squeeze(eval_fd( self.TSpan.Target, self.MeanFd ));

end

