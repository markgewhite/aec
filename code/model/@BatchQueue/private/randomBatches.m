function batches = randomBatches( n, s )
    % Generate a logical arrays defining randomised batches
    arguments
        n       double  % number of observations
        s       double  % batch size
    end

    b = floor( n/s );
    batches = false( n, b );
    bag = 1:n;
    for i = 1:b
        idx = randsample( bag, s );
        batches(idx, i) = true;
        bag = setdiff( bag, idx );
    end

end