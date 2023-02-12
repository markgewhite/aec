function loss = reconRoughnessLoss( XHat, scale, formula )
    % Calculate the roughness penalty from the reconstruction
    arguments
        XHat        {mustBeA( XHat, {'double', 'dlarray'})}
        scale       double = 1
        formula     char {mustBeMember( formula, ...
                    {'3Point', '5Point'} )} = '3Point'

    end

    % calculate the first and second order differences
    switch formula
        case '3Point'
            % using the three-point difference formula
            D1 = 0.5*(XHat(3:end,:,:) - XHat(1:end-2,:,:));
            D2 = 0.5*(D1(3:end,:,:) - D1(1:end-2,:,:));
        case '5Point'
            % using the five-point difference formula
            D1 = (XHat(1:end-4,:,:) - 8*XHat(2:end-3,:,:) ...
                  + 8*XHat(4:end-1,:,:) - XHat(5:end,:,:))/12;
            D2 = (D1(1:end-4,:,:) - 8*D1(2:end-3,:,:) ...
                  + 8*D1(4:end-1,:,:) - D1(5:end,:,:))/12;
    end

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
