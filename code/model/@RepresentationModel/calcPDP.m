function [ dlXCHat, zs, dlZC ] = calcPDP( self, args )
    % Partial Dependence Plot
    % For latent component generation and the auxiliary model
    arguments
        self                RepresentationModel
        args.dlZ            dlarray
        args.mode           char ...
                            {mustBeMember(args.mode, ...
                            {'Full', 'InputOnly', 'OutputOnly'} )} = 'Full' 
        args.dlXC           {mustBeA( args.dlXC, {'dlarray', 'double'} )} = []
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Regular', 'Component'} )} = 'Regular'
        args.nSample        double {mustBeInteger} = 100
        args.maxObs         double = 1000
    end

    if isfield( args, 'dlZ' )
        dlZ = args.dlZ;
    end

    % generate the required Z values from z-scores
    switch args.sampling
        case 'Regular'
            zs = linspace( -2, 2, args.nSample+1 );
        case 'Component'
            zs = linspace( -2, 2, self.NumCompLines );
    end
    K = length(zs);

    if any(strcmp( args.mode,{'Full', 'InputOnly'} ))

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
    
        % take the mean and standard deviation
        dlZMean = mean( dlZ, 2 );
        dlZSD = std( dlZ, [], 2 );
    
        % prepare all inputs for model function to avoid multiple calls
        % set all elements to the mean initially
        dlZC = repmat(dlZMean, 1, self.ZDimAux*K);
        i = 1;
        for d = 1:self.ZDimAux
            for k = 1:K
                % set the dth element to the kth value
                dlZC(d,i) = dlZMean(d) + zs(k)*dlZSD(d);
                i = i + 1;
            end
        end

    else
        dlZC = [];

    end

    if strcmp( args.mode, 'Full' )
        % call the model function to generate a response
        dlXCHat = self.LatentResponseFcn( dlZC );

    elseif strcmp( args.mode, 'OutputOnly' )
        % use the provided XC values
        dlXCHat = args.dlXC;

    else
        dlXCHat = [];
    end
    
    if any(strcmp( args.mode,{'Full', 'OutputOnly'} ))
        % reshape the output
        XDim = size( dlXCHat, 1 );
        dlXCHat = reshape( dlXCHat, XDim, self.ZDimAux, K, [] );
        dlXCHat = permute( dlXCHat, [1 3 2 4] );
    end

end
