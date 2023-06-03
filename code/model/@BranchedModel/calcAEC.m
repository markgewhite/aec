function [ dlXCHat, zs ] = calcAEC( self, args )
    % Generate autoencoder components
    arguments
        self                BranchedModel
        args.dlZ            dlarray
        args.mode           char ...
                            {mustBeMember(args.mode, ...
                            {'Full', 'OutputOnly'} )} = 'Full'
        args.dlXB           cell = []
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Regular', 'Component'} )} = 'Component' 
        args.nSample        double {mustBeInteger} = 100
        args.maxObs         double = 1000
    end

    if isfield( args, 'dlZ' )
        dlZ = args.dlZ;
    end

    if strcmp( args.mode, 'Full' )

        if size(args.dlZ,1) ~= self.ZDim
            % transpose into standard dimensions:
            % 1st=ZDim and 2nd=observations
            dlZ = dlZ';
        end
    
        nObs = size( dlZ, 2 );
        if nObs > args.maxObs
            % data too large - subsample
            subset = randsample( nObs, args.maxObs );
            dlZ = dlZ( :, subset );
        end

    end

    if strcmp( args.mode, 'Full' )
        % run the decoder to generate the components as a cell array
        [ dlXB{1:self.ZDimAux} ] = predict( self.Nets.Decoder, dlZ );

    else
        % use the provided XB cell array
        dlXB = args.dlXB;

    end

    % generate the z-scores
    switch args.sampling
        case 'Regular'
            zs = linspace( -2, 2, args.nSample+1 );
        case 'Component'
            zs = linspace( -2, 2, self.NumCompLines );
    end
    K = length(zs);

    % define the component array
    XDim = size( dlXB{1}, 1 );
    dlXCHat = dlarray( zeros( XDim, K, self.ZDimAux, self.XChannels ) );

    % iterate through the components computing values at z-scores
    for i = 1:self.ZDimAux
        dlXCMean = mean( dlXB{i}, 2 );
        dlXCStd = std( dlXB{i}, [], 2 );
        for k = 1:K
            dlXCHat( :, k, i, : ) = dlXCMean + zs(k)*dlXCStd; 
        end
    end

end
