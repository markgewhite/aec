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
%           Z           : latent encodings sample
% ************************************************************************

function dlXComp = latentComponents( decoder, dlZ )

% faster execution by avoiding the dl array form
Z = extractdata( dlZ );
nComp = size( Z, 1 );
nRepeat = 10;

% compute the mean Z across the batch
ZMean = mean( Z,2 );

% initialise the components' Z codes at the mean
ZComp = repmat( ZMean, 1, nComp*nRepeat+1 );

for j = 1:nRepeat
    for i =1:nComp
    
        % compute the SD for the ith code
        ZStd = std( Z(i,:) );
    
        % adjust the ith randomly about its mean value
        ZComp(i,(j-1)*nComp+i) = ZMean(i) + 2*randn(1,1)*ZStd;
        
    end
end

% generate all the component curves using the decoder
dlZComp = dlarray( ZComp, 'CB');
dlXComp = forward( decoder, dlZComp );

% centre about the mean curve (last curve)
dlXComp = dlXComp-dlXComp(:,end);
dlXComp = dlXComp(:,1:end-1);

end