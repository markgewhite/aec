function [ dlVar, dlCoVar ] = dlVarianceCovariance( dlQ  )
    % Calculate the variance and covariance of a dlarray
    arguments
        dlQ             dlarray
    end
    
    d = size( dlQ, 1 );    

    % remove labels to enable transpose
    dlQ = stripdims(dlQ);

    % calculate the covariance
    dlQCov = cov( dlQ' );

    % extract the diagonal (variance)
    dlVar = dlQCov(1:d+1:end);

    % define the covariance with diagonal zeroed
    dlCoVar = dlQCov - dlVar.*eye(d);

end
