function [ F, zs, ZQ ] = calcPDP( self, dlZ, args )
    % Partial Dependence Plot
    % For latent component generation and the auxiliary model
    arguments
        self                RepresentationModel
        dlZ                 {mustBeA( dlZ, {'dlarray', 'double'} )}
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Regular', 'Component'} )} = 'Regular'
        args.nSample        double {mustBeInteger} = 20
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
    K = args.nSample;
    switch args.sampling
        case 'Regular'
            zs = linspace( -2, 2, K+1 );
        case 'Component'
            zs = linspace( -2, 2, self.NumCompLines );
    end
    K = length(zs);

    % take the mean and standard deviation
    dlZMean = mean( dlZ, 2 );
    dlZSD = std( dlZ, [], 2 );

    ZQ = zeros( self.ZDim, K );
    for d = 1:self.ZDim

        for k = 1:K
   
            % set all elements to the median
            dlZC = dlZMean;
            % set the dth element to the kth value
            dlZC(d) = dlZMean(d) + zs(k)*dlZSD(d);

            % call the model function to generate a response
            YHat = args.modelFcn( dlZC );

            if d==1 && k==1
                % allocate arrays knowing the size of YHat
                if size( YHat, 2 )==1
                    FDim(1) = size( YHat, 1 );
                    FDim(2) = 1;
                    F = zeros( self.ZDim, K, FDim(1) );
                else
                    FDim(1) = size( YHat, 1 );
                    FDim(2) = size( YHat, 2 );
                    F = zeros( self.ZDim, K, FDim(1), FDim(2) );
                end
            end

            % assign YHat to response array
            F( d, k, :, : ) = YHat;

            % store Z
            ZQ( d, k ) = dlZC(d);

        end

    end

end
