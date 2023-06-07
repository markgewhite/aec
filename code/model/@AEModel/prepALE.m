function [ dlZC, A, w ] = prepALE( self, dlZ, args )
    % Accumulated Local Estimation 
    % For latent component generation and the auxiliary model
    arguments
        self                AEModel
        dlZ                 dlarray
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Regular', 'Component'} )} = 'Regular'
        args.nSample        double {mustBeInteger} = 100
        args.maxObs         double = 1000
    end

    % generate the quantiles and required Z values
    switch args.sampling
        case 'Regular'
            zsMid = linspace( -2, 2, args.nSample+1 );
        case 'Component'
            zsMid = linspace( -2, 2, self.NumCompLines );
    end
    zsEdge = [-5, (zsMid(1:end-1)+zsMid(2:end))/2, 5];
    % set the number of bins
    K = length(zsMid);

    % convert to double for quantiles, sort and other functions
    Z = double(extractdata( dlZ ));

    nObs = size( dlZ, 2 );
    if nObs > args.maxObs
        % data too large - subsample
        subset = randsample( nObs, args.maxObs );
        dlZ = dlZ( :, subset );
        Z = Z( :, subset );
        nObs = args.maxObs;
    end

    % take the mean and standard deviation
    dlZMean = mean( dlZ, 2 );
    dlZSD = std( dlZ, [], 2 );

    % generate ZQ and ZQMid values from z-scores
    ZQ = zeros( self.ZDimAux, K+1 );
    ZQMid = zeros( self.ZDimAux, K );
    for d = 1:self.ZDimAux
        for k = 1:K+1
            ZQ( d, k ) = dlZMean(d) + zsEdge(k)*dlZSD(d);
        end
        for k = 1:K
            ZQMid( d, k ) = dlZMean(d) + zsMid(k)*dlZSD(d);
        end
    end

    % identify bin assignments across dimensions
    A = zeros( self.ZDimAux, nObs );
    for d = 1:self.ZDimAux
        [~, orderIdx] = sort( Z(d,:) );
        j = 1;
        for i = orderIdx
            if Z(d,i)<=ZQ(d,j+1)
                A(d,i) = j;
            else
                j = min( j+1, K );
                A(d,i) = j;
            end
        end
    end

    % prepare all inputs for model function to avoid multiple calls
    % set all elements to the mean initially
    ZC1 = repmat(dlZMean, 1, self.ZDimAux*K*nObs);
    ZC2 = ZC1;
    i = 1;
    for d = 1:self.ZDimAux
        for k = 1:K
            rng = i:i+nObs-1;
            % set the dth element to the kth value
            ZC1(d,rng) = ZQ(d, A(d,:));
            ZC2(d,rng) = ZQ(d, A(d,:)+1);
            i = i + nObs;
        end
    end

    dlZC = dlarray( [ ZC1 ZC2 ], 'CB' );

    % set the weights based on the number of occurrences
    % the weights will be used in calcALE
    w = zeros( self.ZDimAux, K );
    for d = 1:self.ZDimAux
        w( d, : ) = histcounts( Z(d,:), unique(ZQ(d,:)) );
    end

end
