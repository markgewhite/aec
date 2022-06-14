function loss = reconLoss( X, XHat, scale )
    % Calculate the reconstruction loss
    arguments
        X           {mustBeA( X, {'double', 'dlarray'})}
        XHat        {mustBeA( XHat, {'double', 'dlarray'})}
        scale       double = 1
    end

    if isa( X, 'dlarray' )
        % dimensions are Time x Channel x Batch
        if size( X, 3 ) == 1
            loss = mean( (X-XHat).^2, [1 2] )/scale;
        else
            loss = mean(mean( (X-XHat).^2, [1 3] )./scale);
        end
    else
        % dimensions are Time x Batch x Channel
        if size( X, 3 ) == 1
            loss = mean( (X-XHat).^2, [1 2] )/scale;
        else
            loss = mean(mean( (X-XHat).^2, [1 2] )./ ...
                                permute(scale, [1 3 2]));
        end
    end

end
