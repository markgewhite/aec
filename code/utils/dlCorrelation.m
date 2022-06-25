function dlR = dlCorrelation( dlZ )
    % Calculate the Pearson correlation on a dlarray
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
    dlD = sqrt(dlTrace( dlR, d, 1 ));
    % and then its transpose 
    % (cannot transpose a dlarray with differently labelled dimensions)
    dlDTranspose = sqrt(dlTrace( dlR, 1, d ));
    % then divide the denominator squared
    dlR = dlR./dlD;
    dlR = dlR./dlDTranspose;

end


function dlVSq = dlVectorSq( dlV, d )
    % Calculate dlV*dlV' (transpose)
    % and preserve the dlarray
    dlVSq = dlV( 1:d, 1:d );
    for i = 1:d
        for j = 1:d
            dlVSq(i,j) = sum( dlV(i,:).*dlV(j,:) );
        end
    end
    dlVSq = gather( dlVSq );

end


function tr = dlTrace( dlV, r, c )
    % Get the trace of a dlarray
    tr = dlV(1:r,1:c);
    for i = 1:max(r,c)
        tr(i) = dlV(i,i);
    end

end