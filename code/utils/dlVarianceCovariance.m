function [ dlVar, dlCoVar ] = dlVarianceCovariance( dlQ )
    % Calculate the variance and covariance of a dlarray
    arguments
        dlQ         dlarray
    end
    
    % remove zero rows
    dlQSum = sum( dlQ, 2 );
    dlQ = dlQ( dlQSum~=0, : );

    [d, n] = size( dlQ );    

    % centre dlZ
    dlQ = dlQ - dlQSum( 1:d )/n;

    % calculate the covariance (dlZ*dlZ')
    dlQSq = dlVectorSq( dlQ, d );

    % extract the trace (variance)
    dlVar = dlTrace( dlQSq, d, 1 );

    % define the covariance with diagonal zeroed
    dlCoVar = dlQSq - dlVar.*eye(d);

end


function dlQSq = dlVectorSq( dlQ, d )
    % Calculate dlV*dlV' (transpose)
    % and preserve the dlarray
    dlQSq = dlQ( 1:d, 1:d );
    for i = 1:d
        for j = 1:d
            dlQSq(i,j) = sum( dlQ(i,:).*dlQ(j,:) );
        end
    end
    dlQSq = gather( dlQSq );

end


function tr = dlTrace( dlQ, r, c )
    % Get the trace of a dlarray
    tr = dlQ(1:r,1:c);
    for i = 1:max(r,c)
        tr(i) = dlQ(i,i);
    end

end