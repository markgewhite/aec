function dlXCHat = calcAEC( self, dlXB, args )
    % Generate autoencoder components
    arguments
        self                BranchedModel
        dlXB                cell
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Random', 'Component'} )} = 'Component'
        args.nSample        double {mustBeInteger} = 100
    end

    % generate the z-scores
    switch args.sampling
        case 'Random'
            zs = 4*(rand(1, args.nSample) - 0.5);
        case 'Component'
            zs = linspace( -2, 2, self.NumCompLines );
    end
    K = length(zs);

    % define the component array
    XDim = size( dlXB{1}, 1 );
    dlXCHat = dlarray( zeros( XDim, K, self.ZDimAux, self.XChannels, 'single' ) );

    % iterate through the components computing values at z-scores
    for i = 1:self.ZDimAux
        dlXCMean = mean( dlXB{i}, 2 );
        dlXCStd = std( dlXB{i}, [], 2 );
        for k = 1:K
            dlXCHat( :, k, i, : ) = dlXCMean + zs(k)*dlXCStd; 
        end
    end

end
