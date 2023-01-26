function [ F, zsMid, ZQMid ] = calcALE( self, dlZ, args )
    % Accumulated Local Estimation 
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
    
    if isa( dlZ, 'dlarray' )
        % convert to double for quantiles, sort and other functions
        Z = double(extractdata( dlZ ));
    else
        if size(dlZ,1) ~= self.ZDim
            % transpose into standard dimensions:
            % 1st=ZDim and 2nd=observations
            dlZ = dlZ';
        end
        Z = dlZ;
    end

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
    ZQ = zeros( self.ZDim, K+1 );
    ZQMid = zeros( self.ZDim, K );
    for d = 1:self.ZDim
        for k = 1:K+1
            ZQ( d, k ) = dlZMean(d) + zsEdge(k)*dlZSD(d);
        end
        for k = 1:K
            ZQMid( d, k ) = dlZMean(d) + zsMid(k)*dlZSD(d);
        end
    end

    % identify bin assignments across dimensions
    A = zeros( self.ZDim, nObs );
    for d = 1:self.ZDim
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

    for d = 1:self.ZDim

        % generate predictions for bin and bin+1
        dlZC1 = dlZ;
        dlZC2 = dlZ;
        dlZC1(d,:) = ZQ(d, A(d,:));
        dlZC2(d,:) = ZQ(d, A(d,:)+1);

        % call the model function to generate responses
        YHat1 = args.modelFcn( dlZC1 );
        YHat2 = args.modelFcn( dlZC2 );
        delta = YHat2 - YHat1;

        if d==1
            % allocate arrays knowing the size of YHat
            if size(delta,3)==1
                FDim(1) = size(delta,1);
                FDim(2) = 1;
                F = zeros( self.ZDim, K, FDim(1) );
                FBin = zeros( K, FDim(1) );
            else
                FDim(1) = size(delta,1);
                FDim(2) = size(delta,2);
                F = zeros( self.ZDim, K, FDim(1), FDim(2) );
                FBin = zeros( K, FDim(1), FDim(2) );
            end
        end

        % subtract the average weighted by number of occurrences
        w = histcounts(Z(d,:), unique(ZQ(d,:)))';

        % calculate means of delta grouped by bin
        % and cumulatively sum
        if FDim(2)==1
            for i = 1:K
                binIdx = (A(d,:)==i);
                if any(binIdx)
                    FBin(i,:) = mean( delta(:,binIdx), 2 );
                end
            end
            FMeanCS = [ zeros(1,FDim(1)); cumsum(FBin) ];
            FMeanMid = (FMeanCS(1:K,:) + FMeanCS(2:K+1,:))/2;
            F( d,:,: ) = FMeanMid - sum(w.*FMeanMid)/sum(w);

        else
            for i = 1:K
                binIdx = (A(d,:)==i);
                if any(binIdx)
                    FBin(i,:,:) = mean( delta(:,:,binIdx), 3 );
                end
            end
            FMeanCS = [ zeros(1,FDim(1),FDim(2)); cumsum(FBin) ];
            FMeanMid = (FMeanCS(1:K,:,:) + FMeanCS(2:K+1,:,:))/2;
            F( d,:,:,: ) = FMeanMid - sum(w.*FMeanMid)/sum(w);

        end

    end

end
