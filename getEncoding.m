% ************************************************************************
% Function: getEncoding
%
% Generate hidden encoding using the encoder network,
% applying the reparameterization trick, if required.
%
% Parameters:
%           dlnetEnc    : trained encoder network
%           XT          : transformed input data
%           setup       : structure of all training/network parameters
%           
% Outputs:
%           dlZ         : latent encoding
%
% ************************************************************************

function dlZ = getEncoding( dlnetEnc, dlXT, setup )

dlZ = predict( dlnetEnc, dlXT );

if setup.variational
    if setup.useVarMean
        dlZ = dlZ( 1:setup.zDim, : );
    else
        dlZ = reparameterize( dlZ );
    end
end

end