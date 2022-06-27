function [ dlR, dlT ] = dlCorrelation( dlZ )
    % Calculate the Pearson correlation on a dlarray
    % and also return the trace (variance)
    arguments
        dlZ         dlarray
    end
    
    % remove zero rows
    dlZSum = sum( dlZ, 2 );
    dlZ = dlZ( dlZSum~=0, : );

    [d, n] = size( dlZ );    

    % centre dlZ
    dlZ = dlZ - dlZSum( 1:d )/n;

    % dlZ*dlZ'       
    dlR = dlVectorSq( dlZ, d );

    % extract the trace
    dlT = sqrt(dlTrace( dlR, d, 1 ));
    % and then its transpose 
    % (cannot transpose a dlarray with differently labelled dimensions)
    dlTTranspose = sqrt(dlTrace( dlR, 1, d ));
    % then divide the denominator squared
    dlR = dlR./dlT;
    dlR = dlR./dlTTranspose;

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