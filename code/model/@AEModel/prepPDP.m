function dlZC = prepPDP( self, dlZ, args )
    % Prepare dlZ for a Partial Dependence Plot component
    arguments
        self                AEModel
        dlZ                 dlarray
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Random', 'Component'} )} = 'Component'
        args.nSample        double {mustBeInteger} = 100
        args.maxObs         double = 1000
    end

    % generate the required Z values from z-scores
    switch args.sampling
        case 'Random'
            zs = 4*(rand(args.nSample) - 0.5);
        case 'Component'
            zs = linspace( -2, 2, self.NumCompLines );
    end
    K = length(zs);

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

end
