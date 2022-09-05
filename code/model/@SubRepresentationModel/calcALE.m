function [F, QMid, ZQMid, offsets ] = calcALE( self, dlZ, args )
    % Accumulated Local Estimation 
    % For latent component generation and the auxiliary model
    arguments
        self                SubRepresentationModel
        dlZ                 {mustBeA( dlZ, {'dlarray', 'double'} )}
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Regular', 'Component'} )} = 'Regular'
        args.nSample        double {mustBeInteger} = 20
        args.maxObs         double = 1000
        args.modelFcn       function_handle
        args.modelFcnArgs   cell = []
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
    K = args.nSample;
    switch args.sampling
        case 'Regular'
            prc = linspace( 0, 1, K+1 );
            offsets = norminv( prc ); % z-scores

        case 'Component'
            q = linspace( 0.05, 0.95, self.NumCompLines );
            prc = sort( [q-0.05 q+0.05] );
            offsets = norminv( q ); % z-scores
    end
    QMid = prc(1:end-1) + diff(prc)/2;
    ZQ = quantile( Z, prc, 2 );
    %ZQ = [ min(Z,[],2) ZQ max(Z,[],2) ];
    K = length(ZQ)-1;

    % identify bin assignments across dimensions
    A = zeros( self.ZDim, nObs );
    for d = 1:self.ZDim
        [~, order] = sort( Z(d,:) );
        j = 1;
        for i = order
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
        if isempty( args.modelFcnArgs )
            YHat1 = args.modelFcn( self, dlZC1 );
            YHat2 = args.modelFcn( self, dlZC2 );
        else
            YHat1 = args.modelFcn( self, dlZC1, args.modelFcnArgs{:} );
            YHat2 = args.modelFcn( self, dlZC2, args.modelFcnArgs{:} );
        end
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
                FBin(i,:) = mean( delta(:,A(d,:)==i), 2 );
            end
            FMeanCS = [ zeros(1,FDim(1)); cumsum(FBin) ];
            FMeanMid = (FMeanCS(1:K,:) + FMeanCS(2:K+1,:))/2;
            F( d,:,: ) = FMeanMid - sum(w.*FMeanMid)/sum(w);

        else
            for i = 1:K
                FBin(i,:,:) = mean( delta(:,:,A(d,:)==i), 3 );
            end
            FMeanCS = [ zeros(1,FDim(1),FDim(2)); cumsum(FBin) ];
            FMeanMid = (FMeanCS(1:K,:,:) + FMeanCS(2:K+1,:,:))/2;
            F( d,:,:,: ) = FMeanMid - sum(w.*FMeanMid)/sum(w);

        end

    end

    ZQMid = ((ZQ(:,1:K) + ZQ(:,2:K+1))/2);

    if strcmp( args.sampling, 'Component' )
        % remove surplus samples
        F = F( :, 1:2:K, :, : );
    end

end
