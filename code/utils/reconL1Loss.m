function loss = reconL1Loss( X, XHat, scale )
    % Calculate the L1 reconstruction loss
    arguments
        X           {mustBeA( X, {'double', 'single', 'dlarray'})}
        XHat        {mustBeA( XHat, {'double', 'single', 'dlarray'})}
        scale       double = 1
    end

    if isa( X, 'dlarray' )
        % dimensions are Time x Channel x Batch
        if size( X, 3 ) == 1
            loss = mean( abs(X-XHat), [1 2] )/scale;
        else
            loss = mean(mean( abs(X-XHat), [1 3] )./scale);
        end
    else
        % dimensions are Time x Batch x Channel
        if size( X, 3 ) == 1
            loss = mean( abs(X-XHat), [1 2] )/scale;
        else
            loss = mean(mean( abs(X-XHat), [1 2] )./ ...
                                permute(scale, [1 3 2]));
        end
    end

end
