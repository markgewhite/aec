function R = latentCodeCorrelation( Z, arg )
    % Calculate the Pearson correlation between latent codes
    arguments
        Z               {mustBeA( Z, {'dlarray', 'double'} )}
        arg.summary     logical = false
    end
    
    if isa( Z, 'dlarray' )
        Z = double(extractdata( Z ))';
    end

    % get the correlation matrix
    R = corr( Z );

    if arg.summary
        % summarise to a single mean value
        R = mean( ( R - eye(size(R,1)) ).^2, 'all' );
    end

end