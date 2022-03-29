% ************************************************************************
% Function: latentComponents
%
% Calculate the funtional components from the latent codes
% using the decoder network.
% For each component, the relevant code is varied randomly about the mean.
% This is more efficient than calculating two components at strict
% 2SD separation from the mean.
%
% Parameters:
%           decoder     : trained decoder network
%           dlZ         : latent encodings sample
%
% ************************************************************************

function dlXComp = latentComponents( decoder, dlZ, nSample )

% number of components
nComp = size( dlZ, 1 );

% compute the mean Z across the batch
dlZMean = mean( dlZ, 2 );
dlZStd = std( dlZ, [], 2 );

% initialise the components' Z codes at the mean
dlZComp = repmat( dlZMean, 1, (nComp+1)*nSample );

for i =1:nComp
    for j = 1:nSample
    
        % adjust the ith randomly about its mean value
        dlZComp(i,(i-1)*nSample+j) = dlZMean(i) + 2*randn*dlZStd(i);
        
    end
end

% generate all the component curves using the decoder
dlXComp = forward( decoder, dlZComp );

% centre about the mean curve (last curve) common to all classes
dlXMean = mean( dlXComp(:,end-nSample+1), 2 );
dlXComp = dlXComp(:,1:end-nSample) - dlXMean;

end