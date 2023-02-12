function loss = reconRoughnessLoss( XHat, scale, h )
    % Calculate the roughness penalty from the reconstruction
    arguments
        XHat        {mustBeA( XHat, {'double', 'dlarray'})}
        scale       double = 1
        h           double {mustBeInteger, ...
                            mustBeGreaterThanOrEqual(h, 1)} = 1
    end

    % calculate the second order derivative
    % using the centred 3-point formula for points away from the boundary
    D2Centred = (XHat(1:end-2*h,:,:) ...
                 - 2*XHat(1+h:end-h,:,:) ...
                 + XHat(1+2*h:end,:,:))/h^2;
    % include the forward difference for the left-hand boundary
    D2Forward = (2*XHat(1,:,:) ...
                 - 5*XHat(1+h,:,:) ...
                 + 4*XHat(1+2*h,:,:) ...
                 - XHat(1+3*h,:,:))/h^3;
    % include the backward difference for the right-hand boundary
    D2Backward = (2*XHat(end,:,:) ...
                 - 5*XHat(end-h,:,:) ...
                 + 4*XHat(end-2*h,:,:) ...
                 - XHat(end-3*h,:,:))/h^3;
    % combine
    D2 = [D2Forward; D2Centred; D2Backward];

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
