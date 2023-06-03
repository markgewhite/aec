function [ dlXCHat, zs ] = calcAEC( self, dlZ, args )
    % Generate autoencoder components
    arguments
        self                BranchedModel
        dlZ                 {mustBeA( dlZ, {'dlarray', 'double'} )}
        args.mode           char ...
                            {mustBeMember(args.mode, ...
                            {'Full', 'OutputOnly'} )} = 'Full'
        args.dlXC           {mustBeA( args.dlXC, {'dlarray', 'double'} )} = []
        args.nSample        double {mustBeInteger} = 100
        args.maxObs         double = 1000
        args.modelFcn       function_handle
    end

    if strcmp( args.mode, 'Full' )

        if size(dlZ,1) ~= self.ZDim
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
        % run the decoder to generate a response as cell array
        dlXB = args.modelFcn( dlZ );

    else
        % use the provided XC values
        dlXB = args.dlXC;

    end

    % generate the z-scores
    switch args.sampling
        case 'Regular'
            zs = linspace( -2, 2, args.nSample+1 );
        case 'Component'
            zs = linspace( -2, 2, self.NumCompLines );
    end
    K = length(zs);





    % reshape the output
    XDim = size( dlXCHat, 1 );
    dlXCHat = reshape( dlXCHat, XDim, self.ZDimAux, K, [] );
    dlXCHat = permute( dlXCHat, [1 3 2 4] );

end
