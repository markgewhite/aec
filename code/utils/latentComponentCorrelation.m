function [ R, C ] = latentComponentCorrelation( XC, nSamples, arg )
    % Calculate the Pearson correlation between latent components
    % and the covariance matrix
    arguments
        XC              {mustBeA( XC, {'dlarray', 'double'} )}
        nSamples        double {mustBeInteger, mustBePositive}
        arg.summary     logical = false
    end

    if isa( XC, 'dlarray' )
        XC = double(extractdata( XC ));
    end

    % put XC into the appropriate structure
    if size( XC, 3 ) == 1
        XC = permute( XC, [1 3 2] );
    end

    [nPts, nChannels, nComp] = size( XC );
    if mod( nComp, 2 )==1
        % odd number: last component must be the mean
        % centre the components
        XC = XC( :,:,1:nComp-1 ) - XC( :,:,end );
        nComp = nComp - 1;
    end

    nComp = nComp/nSamples;

    XC = reshape( XC, nPts, nChannels, nSamples, nComp );

    % calculate the correlation matrics across samples and channels
    R = zeros( nComp, nComp, nChannels);
    C = zeros( nComp, nComp, nChannels);
    for c = 1:nChannels
        for k = 1:nSamples
            XCsample = squeeze( XC(:,c,k,:) );
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
        R = R0;
        C = C0;
    end

end