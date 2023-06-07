function dlXCHat = calcALE( self, dlXC, A, w )
    % Accumulated Local Estimation 
    % For latent component generation and the auxiliary model
    arguments
        self                AEModel
        dlXC                dlarray
        A                   double
        w                   double
    end

    nObs = size( dlXC, 2 )/2;
    delta = dlXC( :, 1:nObs, : ) - dlXC( :, nObs+1:end, : );

    K = size( w, 2 );

    % allocate arrays knowing the size of XCHat
    nPts = size( delta, 1 );
    nChannels = size( delta, 3 );
    dlXCHat = dlarray(zeros( nPts, K, self.ZDimAux, nChannels ));
    XCBin = zeros( nPts, K, nChannels );

    for d = 1:self.ZDimAux

        % calculate means of delta grouped by bin
        % and cumulatively sum
        for k = 1:K
            binIdx = (A(d,:)==k);
            if any(binIdx)
                XCBin(:,k,:) = mean( delta(:,binIdx,:), 2 );
            end
        end

        XCMeanCS = [ zeros( nPts, 1, nChannels ) ...
                    cumsum( XCBin, 2 ) ];

        XCMeanMid = (XCMeanCS(:,1:K,:) + XCMeanCS(:,2:K+1,:))/2;

        dlXCHat( :,:,d,: ) = XCMeanMid - sum(w(d,:).*XCMeanMid)/sum(w(d,:));

    end


end
