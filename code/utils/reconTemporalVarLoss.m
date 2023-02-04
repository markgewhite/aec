function loss = reconTemporalVarLoss( XHat, scale )
    % Calculate the reconstruction temporal variance loss
    arguments
        XHat        {mustBeA( XHat, {'double', 'dlarray'})}
        scale       double = 1
    end

    delta = XHat(2:end,:,:)-XHat(1:end-1,:,:);
    if isa( XHat, 'dlarray' )
        % dimensions are Time x Channel x Batch
        if size( XHat, 3 ) == 1
            loss = mean( delta.^2, [1 2] )/scale;
        else
            loss = mean(mean( delta.^2, [1 3] )./scale);
        end
    else
        % dimensions are Time x Batch x Channel
        if size( XHat, 3 ) == 1
            loss = mean( delta.^2, [1 2] )/scale;
        else
            loss = mean(mean( delta.^2, [1 2] )./ ...
                                permute(scale, [1 3 2]));
        end
    end

end
