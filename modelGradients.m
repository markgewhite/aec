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

loss = dlarray( zeros(6,1), 'CB' );

% --- reconstruction phase ---

% generate latent encodings
[ dlZFake, state.enc ] = forward( dlnetEnc, dlXReal );

% reconstruct curves from latent codes
[ dlXFake, state.dec ] = forward( dlnetDec, dlZFake );

% --- conditional phase ---

% generate the real latent code with a true normal distribution
dlZReal = dlarray( randn( setup.zDim, setup.batchSize ), 'CB' );

% predict if Z is fake using the discriminator
[ dlDFake, state.dis ] = forward( dlnetDis, dlZFake );

% predict if Z is real using the discriminator
dlDReal = forward( dlnetDis, dlZReal );

% --- classification phase ---

% convert the real class
dlCReal = dlarray( onehotencode( setup.cLabels(dlCReal+1), 1 ), 'CB' );

% predict the class from Z using the classifier
[ dlCFake, state.cls ] = forward( dlnetCls, dlZFake );



% --- calculate losses ---

% calculate the reconstruction loss
loss(1) = mean(mean( (dlXFake - dlXReal).^2 ));

% calculate the L2 regularization loss
w = learnables( {dlnetEnc.Learnables, dlnetDec.Learnables} );
loss(2) = setup.weightL2Regularization*mean( sum( w.^2 ) );

% calculate the key-phase component loss
dlXComp = latentComponents( dlnetDec, dlZFake, size(dlCReal,1) );
loss(3) = setup.keyRegularization* ...
                    mean(mean( abs(dlXComp).*compCost( dlXComp ) ));

% calculate the adversarial loss (discriminator)
loss(4) = -setup.disDRegularization* ...
            mean( log(dlDReal + eps) + log(1 - dlDFake + eps) );

% calculate the adversarial loss (encoder)
loss(5) = -setup.disERegularization*mean( log(dlDFake + eps) );

% calculate the classifer loss
loss(6) = setup.clsRegularization*mean(mean( (dlCFake - dlCReal).^2 ));


% calculate gradients
lossEnc = dlarray( sum(loss([1 2 3 5 6])), 'CB' );
lossDec = dlarray( sum(loss([1 2 3])), 'CB' );
lossDis = dlarray( loss(4), 'CB' );
lossCls = dlarray( loss(6), 'CB' );

grad.enc = dlgradient( lossEnc, dlnetEnc.Learnables );
grad.dec = dlgradient( lossDec, dlnetDec.Learnables );
grad.dis = dlgradient( lossDis, dlnetDis.Learnables );
grad.cls = dlgradient( lossCls, dlnetCls.Learnables );


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
