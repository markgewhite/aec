function dlQCorr = dlCorrelation( dlQ )
    % Calculate the correlation of a dlarray
    arguments
        dlQ         dlarray
    end   

    % get the dimensions
    [d, N] = size( dlQ );

    % remove labels to enable transpose
    dlQ = stripdims(dlQ);

    % compute mean and standard deviation along rows
    dlMu = mean( dlQ, 2);
    dlSigma = std( dlQ, [], 2);
    
    if any(dlSigma==0)
        % avoid divide by zero and exit
        dlQCorr = dlarray( zeros(d, d), 'CB' );
    
    else
        % compute centered data
        dlQC = dlQ - dlMu;
        
        % compute correlation matrix
        dlQCorr = (dlQC*dlQC')./((N-1)*(dlSigma*dlSigma'));
    
        % remove the diagonal
        dlQCorr = dlQCorr - eye(d);
    end
    
end