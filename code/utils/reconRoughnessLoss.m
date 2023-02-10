function loss = reconRoughnessLoss( XHat, scale )
    % Calculate the roughness penalty from the reconstruction
    arguments
        XHat        {mustBeA( XHat, {'double', 'dlarray'})}
        scale       double = 1
    end

    % calculate the first and second order differences
    D1 = XHat(2:end,:,:) - XHat(1:end-1,:,:);
    D2 = D1(2:end,:,:) - D1(1:end-1,:,:);

    if isa( XHat, 'dlarray' )
        % dimensions are Time x Channel x Batch
        if size( XHat, 3 ) == 1
            loss = mean( D2.^2, [1 2] )/scale;
        else
            loss = mean(mean( D2.^2, [1 3] )./scale);
        end
    else
        % dimensions are Time x Batch x Channel
        if size( XHat, 3 ) == 1
            loss = mean( D2.^2, [1 2] )/scale;
        else
            loss = mean(mean( D2.^2, [1 2] )./ ...
                                permute(scale, [1 3 2]));
        end
    end

end
