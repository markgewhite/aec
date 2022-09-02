function [ R, C ] = latentComponentCorrelation( XC, arg )
    % Calculate the Pearson correlation between latent components
    % and the covariance matrix
    arguments
        XC              double
        arg.summary     logical = false
    end

    [nPts, nSamples, nComp, nChannels] = size( XC );

    % calculate the correlation matrics across samples and channels
    R = zeros( nComp, nComp, nChannels);
    C = zeros( nComp, nComp, nChannels);
    for c = 1:nChannels
        for k = 1:nSamples
            XCsample = squeeze( XC(:,k,:,c) );
            R(:,:,c) = R(:,:,c) + corr( XCsample );
            C(:,:,c) = C(:,:,c) + cov( XCsample );
        end
    end
    % average over the samples
    R = R/nSamples;
    C = C/nSamples;

    if arg.summary
        % summarise to a single mean value
        R0 = zeros( nChannels, 1 );
        C0 = zeros( nChannels, 1 );
        for c = 1:nChannels
            R0(c) = sum( ( squeeze(R(:,:,c)) - eye(nComp) ).^2, 'all' ) ...
                            /(nComp*(nComp-1));
            C0(c) = sum( ( squeeze(C(:,:,c)) - eye(nComp) ).^2, 'all' ) ...
                            /(nComp*(nComp-1));
        end
        R = mean(R0);
        C = mean(C0);
    end

end