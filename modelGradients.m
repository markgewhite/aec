% ************************************************************************
% Function: modelGradients
%
% Compute the model gradients with a discriminator
%
% Parameters:
%           dlnetEnc    : encoder network
%           dlnetDec    : decoder network
%           dlnetCls    : classifier network
%           dlXReal     : training data (batch)
%           dlCReal     : training data (batch)
%           setup       : training parameters
%           
% Outputs:
%           grad        : gradients for updating network parameters
%           state       : network training states
%           loss        : computed loss functions
%
% ************************************************************************

function [  grad, state, loss ] = ...
                                modelGradients( ...
                                                dlnetEnc, ...
                                                dlnetDec, ...
                                                dlnetCls, ...
                                                dlXReal, ...
                                                dlCReal, ...
                                                setup )

loss = dlarray( zeros(6,1), 'CB' );

% --- reconstruction phase ---

% generate latent encodings
[ dlZFake, state.enc ] = forward( dlnetEnc, dlXReal );

% reconstruct curves from latent codes
[ dlXFake, state.dec ] = forward( dlnetDec, dlZFake );

% --- classification phase ---

if ~setup.pretraining
    % convert the real class
    dlCReal = dlarray( onehotencode( setup.cLabels(dlCReal+1), 1 ), 'CB' );
    
    % predict the class from Z using the classifier
    [ dlCFake, state.cls ] = forward( dlnetCls, dlZFake );
end


% --- calculate losses ---

% calculate the reconstruction loss
loss(1) = mean(mean( (dlXFake - dlXReal).^2 ));

if setup.l2regularization
    % calculate the L2 regularization loss
    w = learnables( {dlnetEnc.Learnables, dlnetDec.Learnables} );
    loss(2) = setup.weightL2Regularization*mean( sum( w.^2 ) );
end

if setup.orthogonal
    % calculate the orthogonal loss to encourage mutual independence
    ZFake = extractdata( dlZFake );
    orth = ZFake*ZFake';
    loss(3) = setup.orthRegularization* ...
                sqrt(sum(orth.^2,'all') - sum(diag(orth).^2))/ ...
                sum(ZFake.^2,'all');
end

if setup.keyCompLoss
    % calculate the key-phase component loss
    dlXComp = latentComponents( dlnetDec, dlZFake, size(dlCReal,1) );
    loss(4) = setup.keyRegularization* ...
                        mean(mean( abs(dlXComp).*compCost( dlXComp ) ));
end

if ~setup.pretraining
    % calculate the classifer loss
    loss(5) = setup.clsRegularization*mean(mean( (dlCFake - dlCReal).^2 ));
    
    % calculate the cluster loss
    loss(6) = setup.cluRegularization* ...
                    clusterLoss( dlZFake, dlCFake, setup.cLabels );
end

% calculate gradients
lossEnc = dlarray( sum(loss([1 2 3 5 6]) ), 'CB' );
lossDec = dlarray( sum(loss([1 2]) ), 'CB' );

grad.enc = dlgradient( lossEnc, dlnetEnc.Learnables, 'RetainData', true );
grad.dec = dlgradient( lossDec, dlnetDec.Learnables, 'RetainData', true );

if ~setup.pretraining
    lossCls = dlarray( loss(5), 'CB' );
    grad.cls = dlgradient( lossCls, dlnetCls.Learnables );
else
    grad.cls = 0;
end

end


function L = learnables( nets )

    nNets = length( nets );
    n = 0;
    % count the number of learnables
    for i = 1:nNets
        for j = 1:length(nets{i}.Value)
            n = n + numel(nets{i}.Value{j});
        end
    end
    
    L = zeros( n, 1 );
    % concatenate the leanables into a flattened array
    k = 1;
    for i = 1:nNets
        for j = 1:length(nets{i}.Value)
            w = nets{i}.Value{j};
            w = reshape( w, numel(w), 1 );
            L(k:k+length(w)-1) = w;
            k = k+length(w);
        end
    end

end


function XCost = compCost( dlXComp )

[l, n] = size( dlXComp );

XAbs = extractdata(abs( dlXComp ));
XCost = zeros( size(dlXComp) );

for j = 1:n
    % find the peak value
    [pk, pkloc] = max( XAbs(:,j) );
    threshold = 0.1*pk;
    % assign cost to the right
    i = pkloc;
    while i<l
        i = i+1;
        if XAbs(i,j) < threshold
            break
        end
    end
    XCost( i:end, j ) = 1;
    % assign cost to the right
    i = pkloc;
    while i>1
        i = i-1;
        if XAbs(i,j) < threshold
            break
        end
    end
    XCost( 1:i, j ) = 1;
end

end


function L = clusterLoss( dlZ, dlC, labels )

Z = extractdata( dlZ )';
[ nObs, nDim ] = size( Z );

C = extractdata( dlC )';
nGrps = length( labels );

% standardise
Z = (Z - mean(Z))./std( Z );

% calculate the group centroids and deviations within the groups
grpMean = zeros( nGrps-1, nDim );
dist = zeros( nObs, nGrps-1 );
lossW = 0;
for i = 2:nGrps
    % compute the weighted group centroid
    grpMean( i, : ) = sum( C( :,i ).*Z )/sum( C(:,i) );
    % calculate the Euclidean distances to this centroid
    dist( :, i ) = sum( (Z-grpMean(i,:)).^2, 2 );
    % add up the weighted distances
    lossW = lossW + sum( C(:,i).*dist(:,i) );
end
% take the mean square diffence
lossW = lossW/(nGrps*nObs);

% calculate between-groups loss (mutual separation)
if nGrps>1
    lossB = -meanSqDiffBtwAll( grpMean( 2:end,: ) );
else
    lossB = 0;
end

L = lossB + lossW;


end


function d = meanSqDiffBtwAll( x ) 

n = size( x, 1 );
d = 0;
for i = 1:n
    for j = i+1:n
      d = d + sum( (x(i,:)-x(j,:)).^2 );
    end
end
d = 0.5*d/(n*(n-1));

end