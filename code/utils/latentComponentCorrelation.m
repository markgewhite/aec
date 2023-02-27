function [ R, C ] = latentComponentCorrelation( XC, arg )
    % Calculate the Pearson correlation between latent components
    % and the covariance matrix
    arguments
        XC              double
        arg.summary     logical = false
    end

    [~, nSamples, nComp, nChannels] = size( XC );

    % remove the centre-line sample if zero
    midSample = fix(nSamples/2)+1;
    if sum(XC(:,midSample,:,:))==0
        XC(:,midSample,:,:) = [];
        nSamples = nSamples - 1;
    end

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
        % summarise to the average of the off-diagonal elements
        R0 = zeros( nChannels, 1 );
        C0 = zeros( nChannels, 1 );
        for c = 1:nChannels
            R0(c) = sum( abs( squeeze(R(:,:,c)) - eye(nComp) ), 'all' ) ...
                            /(nComp*(nComp-1));
            Cs = squeeze(C(:,:,c));
            C0(c) = sum( abs( Cs- Cs(1:nComp+1:end).*eye(nComp) ), 'all' ) ...
                            /(nComp*(nComp-1));
        end
        R = mean(R0);
        C = mean(C0);
    end

end