% ************************************************************************
% Function: modelGradients
%
% Compute the model gradients with a discriminator
%
% Parameters:
%           dlnetEnc    : encoder network
%           dlnetDec    : decoder network
%           dlnetDis    : discriminator network
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
                                                dlnetDis, ...
                                                dlnetCls, ...
                                                dlXReal, ...
                                                dlCReal, ...
                                                setup )

% --- reconstruction phase ---

% generate latent encodings
[ dlEncOutput, state.enc ] = forward( dlnetEnc, dlXReal );

if setup.variational
    % behave as a variational autoencoder
    [ dlZFake, dlMu, dlLogVar ] = reparameterize( dlEncOutput, setup.nDraw );
    dlXReal = repmat( dlXReal, 1, setup.nDraw );
    dlCReal = repmat( dlCReal, setup.nDraw, 1 );
else
    dlZFake = dlEncOutput;
end

% reconstruct curves from latent codes
[ dlXFake, state.dec ] = forward( dlnetDec, dlZFake );

% calculate the reconstruction loss
loss.recon = mean(mean( (dlXFake - dlXReal).^2 ));

if setup.adversarial   
    % predict authenticity from real Z using the discriminator
    dlZReal = dlarray( randn( setup.zDim, setup.batchSize ), 'CB' );
    dlDReal = forward( dlnetDis, dlZReal );
    
    % predict authenticity from fake Z
    [ dlDFake, state.dis ] = forward( dlnetDis, dlZFake );
    
    % discriminator loss for Z
    loss.dis = -setup.reg.dis* ...
                    0.5*mean( log(dlDReal + eps) + log(1 - dlDFake + eps) );
    loss.gen = -setup.reg.gen* ...
                    mean( log(dlDFake + eps) );
else
    loss.dis = 0;
    loss.gen = 0;
    state.dis = [];
end

if setup.variational && ~setup.adversarial
    % calculate the variational loss
    loss.var = -setup.reg.beta* ...
        0.5*mean( sum(1 + dlLogVar - dlMu.^2 - exp(dlLogVar)) );
else
    loss.var = 0;
end

if setup.wasserstein
    % calculate the maximum mean discrepancy loss
    dlZReal = dlarray( randn( setup.zDim, setup.batchSize ), 'CB' );
    loss.mmd = mmdLoss( dlZFake, dlZReal, setup.mmd );
else
    loss.mmd = 0;
end


% --- classification phase ---

if ~setup.pretraining
    % predict the class from Z
    switch setup.classifier
        case 'Network'
            dlCReal = dlarray( ...
                onehotencode( setup.cLabels(dlCReal+1), 1 ), 'CB' );
            [ dlCFake, state.cls ] = forward( dlnetCls, dlZFake );
            loss.cls = setup.reg.cls* ...
                    crossentropy( dlCFake, dlCReal, ...
                                  'TargetCategories', 'Independent' ); 

        otherwise
            ZFake = double(extractdata( dlZFake ))';
            CReal = double(extractdata( dlCReal ));
            switch setup.classifier
                case 'Fisher'
                    modelCls = fitcdiscr( ZFake, CReal );
                case 'SVM'
                    modelCls = fitcecoc( ZFake, CReal );
            end
            loss.cls = setup.reg.cls*resubLoss( modelCls );
            CFake = predict( modelCls, ZFake );
            dlCFake = dlarray( ...
                    onehotencode( setup.cLabels(CFake+1), 1 ), 'CB' );

    end

else
    loss.cls = 0;
end


% --- calculate other losses ---

if setup.l2regularization
    % calculate the L2 regularization loss
    w = learnables( {dlnetEnc.Learnables, dlnetDec.Learnables} );
    loss.wl2 = setup.reg.wl2*mean( sum( w.^2 ) );
else
    loss.wl2 = 0;
end

if setup.orthogonal
    % calculate the orthogonal loss to encourage mutual independence
    ZFake = extractdata( dlZFake );
    orth = ZFake*ZFake';
    loss.orth = setup.reg.orth* ...
                    sqrt(sum(orth.^2,'all') - sum(diag(orth).^2))/ ...
                    sum(ZFake.^2,'all');
else
    loss.orth = 0;
end

if setup.keyCompLoss
    % calculate the key-phase component loss
    dlXComp = latentComponents( dlnetDec, dlZFake, size(dlCReal,1) );
    loss.comp = setup.reg.comp* ...
                        mean(mean( abs(dlXComp).*compCost( dlXComp ) ));
else
    loss.comp = 0;
end

if setup.clusterLoss && ~setup.pretraining
    % calculate the cluster loss
    loss.clust = setup.reg.clust* ...
                clusterLoss( dlZFake, dlCFake, setup.cLabels );
else
    loss.clust = 0;
end


% --- calculate gradients ---
lossEnc = loss.recon + loss.gen + loss.var + loss.cls + loss.comp;
lossDec = loss.recon + loss.dis;

grad.enc = dlgradient( lossEnc, dlnetEnc.Learnables, 'RetainData', true );
grad.dec = dlgradient( lossDec, dlnetDec.Learnables, 'RetainData', true );

if setup.adversarial
    grad.dis = dlgradient( loss.dis, dlnetDis.Learnables, 'RetainData', true );
else
    grad.dis = 0;
end

if ~setup.pretraining
    grad.cls = dlgradient( loss.cls, dlnetCls.Learnables );
else
    grad.cls = 0;
end

loss = struct2array( loss );

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
% compute the weighted group centroid
for i = 2:nGrps
    grpMean( i, : ) = sum( C( :,i ).*Z )/sum( C(:,i) );
end

dist = zeros( nObs, nGrps-1 );
lossW = 0;
for i = 2:nGrps
    % calculate the Euclidean distances to ith centroid
    dist( :, i ) = sum( (Z-grpMean(i,:)).^2, 2 );
    for j = 2:nGrps
        % weighted distances to centroid
        wdist = sum( C(:,j).*dist(:,i) );
        if j==i 
            % add 
            lossW = lossW + wdist;
        else
            lossW = lossW - wdist;
        end
    end
end
% take the mean square diffence
lossW = lossW/(nGrps*nGrps*nObs);

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