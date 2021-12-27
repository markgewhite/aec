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
%           nClass      : number of classes
% ************************************************************************

function dlXComp = latentComponents( decoder, dlZ, nClass )

% faster execution by avoiding the dl array form
Z = extractdata( dlZ );

% number of components
nComp = size( Z, 1 );

% ignore the null class
nClass = nClass - 1;

% compute the mean Z across the batch
ZMean = mean( Z,2 );
ZStd = std( Z, [], 2 );

% initialise the components' Z codes at the mean
ZComp = repmat( ZMean, 1, (nComp+1)*nClass );

% initialise the classes with hot encodes
C = [ zeros( 1, nClass ); eye( nClass ) ];
dlCComp = dlarray( repmat( C, 1, nComp+1 ), 'CB' );

for i =1:nComp
    for j = 1:nClass
    
        % adjust the ith randomly about its mean value
        ZComp(i,(i-1)*nClass+j) = ZMean(i) + 2*randn(1,1)*ZStd(i);
        
    end
end

% generate all the component curves using the decoder
dlZComp = dlarray( ZComp, 'CB');
dlXComp = forward( decoder, [dlZComp; dlCComp] );

% centre about the mean curve (last curve) common to all classes
dlXMean = mean( dlXComp(:,end-nClass+1), 2 );
dlXComp = dlXComp(:,1:end-nClass) - dlXMean;

end