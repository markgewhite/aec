function dlXCHat = calcAEC( self, dlZ, dlXB, args )
    % Generate autoencoder components
    arguments
        self                BranchedModel
        dlZ                 dlarray
        dlXB                cell
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Random', 'Component'} )} = 'Component'
        args.nSample        double {mustBeInteger} = 100
    end

    % switch to single from dlarray for speed
    % tracing is not relevant here
    Z = single(extractdata( dlZ ));

    % generate the z-scores
    switch args.sampling
        case 'Random'
            zsMid = sort(4*(rand(1, args.nSample) - 0.5));
        case 'Component'
            zsMid = linspace( -2, 2, self.NumCompLines );
    end
    zsEdge = [-3, (zsMid(1:end-1)+zsMid(2:end))/2, 3];
    K = length(zsMid);

    % take the mean and standard deviation of the latent codes
    ZMean = mean( Z, 2 );
    ZSD = std( Z, [], 2 );

    % generate ZQ and ZQMid values from z-scores
    ZQEdge = zeros( self.ZDimAux, K+1 );
    ZQMid = zeros( self.ZDimAux, K );
    for d = 1:self.ZDimAux
        for k = 1:K
            ZQEdge( d, k ) = ZMean(d) + zsEdge(k)*ZSD(d);
            ZQMid( d, k ) = ZMean(d) + zsMid(k)*ZSD(d);
        end
        ZQEdge( d, K+1 ) = ZMean(d) + zsEdge(K+1)*ZSD(d);
    end

    % identify bin assignments across dimensions
    nObs = size( Z, 1 );
    A = zeros( self.ZDimAux, nObs );
    for d = 1:self.ZDimAux
        [~, orderIdx] = sort( Z(d,:) );
        j = 1;
        for i = orderIdx
            if Z(d,i)<=ZQEdge(d,j+1)
                A(d,i) = j;
            else
                j = min( j+1, K );
                A(d,i) = j;
            end
        end
    end

    % define the component array
    XDim = size( dlXB{1}, 1 );
    dlXCHat = dlarray( zeros( XDim, K, self.ZDimAux, self.XChannels, 'single' ) );

    % iterate through the components 
    for d = 1:self.ZDimAux
        zeroCols = [];
        % iterate across bins (conditional mean)
        for k = 1:K
            inBin = (A(d,:)==k);
            if any(inBin)
                dlXCHat( :, k, d, : ) = mean( dlXB{d}( :, inBin, : ), 2 );
            else
                zeroCols = [zeroCols, k]; %#ok<AGROW> 
            end
        end
        % interpolate to fill in gaps
        if ~isempty(zeroCols)
            nonZeroCols = setdiff(1:K, zeroCols);
            if ~isempty(nonZeroCols)
                x = zsMid( nonZeroCols );
                v = dlXCHat(:, nonZeroCols, d, :);
                xq = zsMid( zeroCols );
                for c = 1:self.XChannels
                    for i = 1:XDim
                        dlXCHat(i, zeroCols, d, c) = interp1(x, v(i,:,c), xq, 'linear', 'extrap');
                    end
                end
            end
        end

    end


end
