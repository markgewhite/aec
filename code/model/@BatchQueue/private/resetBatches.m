function batches = resetBatches( n, s, r )
    % Generate a logical arrays defining predetermined batches
    arguments
        n       double  % number of observations
        s       double  % batch size
        r       double  % reset order
    end

    b = floor( n/s );
    batches = false( n, b );
    idx = 1:s;
    for i = 1:b       
        batches(r(idx), i) = true;
        idx = idx+s;
    end

end