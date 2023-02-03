function loss = reconTemporalL1Loss( X, XHat, scale )
    % Calculate the L1 reconstruction loss over the time domain
    arguments
        X           {mustBeA( X, {'double', 'dlarray'})}
        XHat        {mustBeA( XHat, {'double', 'dlarray'})}
        scale       double = 1
    end

    if isa( X, 'dlarray' )
        % dimensions are Time x Channel x Batch
        if size( X, 3 ) == 1
            loss = mean( abs(X-XHat), 2 )/scale;
        else
            loss = mean(mean( abs(X-XHat), 3 )./scale);
        end
    else
        % dimensions are Time x Batch x Channel
        if size( X, 3 ) == 1
            loss = squeeze(mean( abs(X-XHat), 2 )/scale);
        else
            loss = squeeze(mean( abs(X-XHat), 2 )./ ...
                                permute(scale, [1 3 2]));
        end
    end

end