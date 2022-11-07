function [ dlVar, dlCoVar ] = dlVarianceCovariance( dlQ )
    % Calculate the variance and covariance of a dlarray
    arguments
        dlQ         dlarray
    end
    
    % remove zero rows
    dlQSum = sum( dlQ, 2 );
    [d, n] = size( dlQ );    

    % centre dlZ
    dlQ = dlQ - dlQSum( 1:d )/n;

    % remove labels to enable transpose
    dlQ = stripdims(dlQ);

    % calculate the covariance (dlQ*dlQ')
    dlQSq = dlQ*dlQ';

    % extract the trace (variance)
    dlVar = dlQSq(1:d+1:end);

    % define the covariance with diagonal zeroed
    dlCoVar = dlQSq - dlVar.*eye(d);

end
