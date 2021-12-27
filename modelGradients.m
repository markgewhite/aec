% ************************************************************************
% Function: modelGradients
%
% Compute the model gradients
%
% Parameters:
%           dlnetEnc    : encoder network
%           dlnetDec    : decoder network
%           dlXReal     : training data (batch)
%           setup       : training parameters
%           
% Outputs:
%           grad        : gradients for updating network parameters
%           state       : network training states
%           loss        : computed loss functions
%           score       : computed scores for tracking progress
%
% ************************************************************************

function [  grad, state, loss, score ] = ...
                                modelGradients( ...
                                                dlnetEnc, ...
                                                dlnetDec, ...
                                                dlXReal, ...
                                                setup )

loss = dlarray( zeros(4,1), 'CB' );

% --- reconstruction phase ---

% generate latent encodings
[ dlZFake, state.enc ] = forward( dlnetEnc, dlXReal );

% reconstruct curves from latent codes
[ dlXFake, state.dec ] = forward( dlnetDec, dlZFake );

% calculate the reconstruction loss
loss(1) = mean(mean( (dlXFake - dlXReal).^2 ));

% calculate the L2 regularization loss
w = learnables( {dlnetEnc.Learnables, dlnetDec.Learnables} );

loss(2) = setup.weightL2Regularization*mean( sum( w.^2 ) );

% calculate the component roughness loss
dlXComp = latentComponents( dlnetDec, dlZFake );
XComp = double(extractdata(dlXComp));
XCompD2 = diff( XComp, 2 );
loss(3) = setup.curveD2Regularization*mean( sum( XCompD2.^2 ) );

% calculate the turning points loss
XCompD1 = diff( XComp, 1 );
tp = XCompD1(1:end-1,:).*XCompD1(2:end,:);
nTP = sum( tp<0 );
loss(4) = setup.tpRegularization*mean( nTP );

% calculate the key-phase component loss
loss(5) = setup.keyRegularization* ...
                    mean(mean( abs(dlXComp).*compCost( dlXComp ) ));

% calculate the component correlation loss
loss(6) = setup.corrRegularization*mean( abs(innerProduct( dlXComp )) );


% calculate gradients
totalLoss = dlarray( sum(loss(1:6)), 'CB' );
grad.enc = dlgradient( totalLoss, dlnetEnc.Learnables );
grad.dec = dlgradient( totalLoss, dlnetDec.Learnables );
                               
% Calculate the scores
score = 0;

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


function XInProdSum = innerProduct( dlXComp )

[l, n] = size( dlXComp );

XComp = extractdata( dlXComp );
XComp = (XComp - mean(XComp,2))./std(XComp,[],2);
XInProd = zeros( l, n*(n-1)/2 );
k = 0;
for i = 1:n
    for j = i+1:n
        k = k+1;
        XInProd(:,k) = XComp(:,i).*XComp(:,j);
    end
end
XInProdSum = mean( XInProd );

end