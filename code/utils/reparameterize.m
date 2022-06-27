function [dlZ, dlMu, dlLogVar] = reparameterize( dlOutput, nDraws )
    % Perform the reparameterization trick
    % Draw from the Gaussian distribution defined by the 
    % mean and log variance from the output
    arguments
        dlOutput   dlarray
        nDraws     double  {mustBeInteger,mustBePositive} = 1
    end
    
    ZDim = size( dlOutput, 1 )/2;
    
    %dlMu = repmat( dlOutput( 1:ZDim, : ), 1, nDraws );
    dlMu = dlOutput( 1:ZDim, : );

    %dlLogVar = repmat( dlOutput( ZDim+1:end, : ), 1, nDraws );
    dlLogVar = dlOutput( ZDim+1:end, : );

    dlSigma = exp( 0.5*dlLogVar );
    
    dlEpsilon = randn( size(dlSigma) );
    
    dlZ = dlMu + dlEpsilon.*dlSigma;
    
end