% ************************************************************************
% Function: reparameterize
%
% Obtain Z from dlnetEnc output, and mu and sigma
%
% Parameters:
%           dlEncOutput    : encoder network output
%           nDraw          : number of draws from distribution per point
%           
% Outputs:
%           dlZ            : computed Z
%           dlMu           : mean
%           dlLogVar       : sigma ^ 2
%
% ************************************************************************

function [dlZ, dlMu, dlLogVar] = reparameterize( dlEncOutput, nDraw )

if nargin==1
    nDraw = 1;
end

zDim = size( dlEncOutput, 1 )/2;

dlMu = repmat( dlEncOutput( 1:zDim, : ), 1, nDraw );

dlLogVar = repmat( dlEncOutput( zDim+1:end, : ), 1, nDraw );
dlSigma = exp( 0.5*dlLogVar );

dlEpsilon = randn( size(dlSigma) );

dlZ = dlMu + dlEpsilon.*dlSigma;

end