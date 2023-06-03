function [ F, zsMid, ZQMid ] = calcALE( self, args )
    % Accumulated Local Estimation 
    % For latent component generation and the auxiliary model
    arguments
        self                RepresentationModel
        args.dlZ            dlarray
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Regular', 'Component'} )} = 'Regular'
        args.nSample        double {mustBeInteger} = 100
        args.maxObs         double = 1000
    end
    
    if isfield( args, 'dlZ' )
        dlZ = args.dlZ;
    end

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

    % generate the quantiles and required Z values
    switch args.sampling
        case 'Regular'
            zsMid = linspace( -2, 2, args.nSample );
        case 'Component'
            zsMid = linspace( -2, 2, self.NumCompLines );
    end
    zsEdge = [-5, (zsMid(1:end-1)+zsMid(2:end))/2, 5];
    % set the number of bins
    K = length(zsMid);

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
    dlZC1 = dlarray( repmat(dlZMean, 1, self.ZDimAux*K*nObs), 'CB' );
    dlZC2 = dlZC1;
    i = 1;
    for d = 1:self.ZDimAux
        for k = 1:K
            rng = i:i+nObs-1;
            % set the dth element to the kth value
            dlZC1(d,rng) = ZQ(d, A(d,:));
            dlZC2(d,rng) = ZQ(d, A(d,:)+1);
            i = i + nObs;
        end
    end

    % call the model function to generate responses
    dlXCHat1 = self.LatentResponseFcn( dlZC1 );
    dlXCHat2 = self.LatentResponseFcn( dlZC2 );
    delta = dlXCHat2 - dlXCHat1;

    % allocate arrays knowing the size of XCHat
    nPts = size( delta, 1 );
    nChannels = size( delta, 3 );
    F = zeros( nPts, K, self.ZDimAux, nChannels );
    FBin = zeros( nPts, K, nChannels );
    if isa( dlXCHat1, 'dlarray' )
        % make it a dlarray without labels
        F = dlarray( F );
    end

    for d = 1:self.ZDimAux

        % subtract the average weighted by number of occurrences
        w = histcounts(Z(d,:), unique(ZQ(d,:)));

        % calculate means of delta grouped by bin
        % and cumulatively sum
        for k = 1:K
            binIdx = (A(d,:)==k);
            if any(binIdx)
                FBin(:,k,:) = mean( delta(:,binIdx,:), 2 );
            end
        end
        FMeanCS = [ zeros( nPts, 1, nChannels ) ...
                    cumsum( FBin, 2 ) ];
        FMeanMid = (FMeanCS(:,1:K,:) + FMeanCS(:,2:K+1,:))/2;
        F( :,:,d,: ) = FMeanMid - sum(w.*FMeanMid)/sum(w);

    end

end
