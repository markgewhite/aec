function [ XC, XMean, zs ] = calcLatentComponents( self, Z, args )
    % Calculate the functional principal components
    % Or use the response function
    arguments
        self                PCAModel
        Z                   double
        args.maxObs         double {mustBeInteger} = 500
        args.responseFcn    function_handle
    end

    if ~strcmp( self.ComponentType, 'FPC' )
        % generate a PDP or ALE type component
        argsCell = namedargs2cell( args );
        [XC, XMean, zs] = ...
            calcLatentComponents@RepresentationModel( self, Z', argsCell{:} );
        return
    end

    % compute the components
    nSample = self.NumCompLines;
    % set z-score levels
    zs = linspace( -2, 2, nSample );

    % calculate the standard deviation
    ZSD = std( Z );
    
    % XC structure: Points, Samples, Components, Channels
    XC = zeros( length(self.PCATSpan), nSample, self.ZDim, self.XChannels );

    for i =1:self.ZDim
        FPC = squeeze(eval_fd( self.PCATSpan, self.CompFd(i) ));
        for c = 1:self.XChannels
            for j = 1:nSample
                XC(:,j,i,c) = zs(j)*ZSD(1,i,c)*FPC(:,c);
            end
        end
    end

    XMean = squeeze(eval_fd( self.PCATSpan, self.MeanFd ));

end

