function [ dlXCHat, zs, dlZC ] = calcPDP( self, dlZ, args )
    % Partial Dependence Plot
    % For latent component generation and the auxiliary model
    arguments
        self                RepresentationModel
        dlZ                 {mustBeA( dlZ, {'dlarray', 'double'} )}
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Regular', 'Component'} )} = 'Regular'
        args.nSample        double {mustBeInteger} = 100
        args.maxObs         double = 1000
        args.modelFcn       function_handle
    end

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

    % generate the required Z values from z-scores
    switch args.sampling
        case 'Regular'
            zs = linspace( -2, 2, args.nSample+1 );
        case 'Component'
            zs = linspace( -2, 2, self.NumCompLines );
    end
    K = length(zs);

    % take the mean and standard deviation
    dlZMean = mean( dlZ, 2 );
    dlZSD = std( dlZ, [], 2 );

    % prepare all inputs for model function to avoid multiple calls
    dlZC = dlarray( zeros(self.ZDim, self.ZDim*K), 'CB' );
    i = 1;
    for d = 1:self.ZDim
        for k = 1:K
            % set all elements to the median
            dlZC(:,i) = dlZMean;
            % set the dth element to the kth value
            dlZC(d,i) = dlZMean(d) + zs(k)*dlZSD(d);
            i = i + 1;
        end
    end

    % call the model function to generate a response
    dlXCHat = args.modelFcn( dlZC );

    dlXCHat = reshape( dlXCHat, [], K, self.ZDim );

end
