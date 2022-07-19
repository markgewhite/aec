function [ R, C ] = latentCodeCorrelation( Z, arg )
    % Calculate the Pearson correlation between latent codes
    % and the covariance matrix
    arguments
        Z               {mustBeA( Z, {'dlarray', 'double'} )}
        arg.summary     logical = false
    end
    
    if isa( Z, 'dlarray' )
        Z = double(extractdata( Z ))';
    end

    % ensure Z is two dimensional
    Z = reshape( Z, size(Z,1), [] );

    % get the correlation and covariance matrices
    R = corr( Z );
    C = cov( Z );

    if arg.summary
        % summarise to a single mean square value
        d = size(R,1);
        R = sum( ( R - eye(d) ).^2, 'all' )/(d*(d-1));
        C = sum( ( C - eye(d) ).^2, 'all' )/(d*(d-1));
    end

end