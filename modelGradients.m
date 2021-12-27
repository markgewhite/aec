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

loss = dlarray( zeros(3,1), 'CB' );

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

% calculate the key-phase component loss
dlXComp = latentComponents( dlnetDec, dlZFake );
loss(3) = setup.keyRegularization* ...
                    mean(mean( abs(dlXComp).*compCost( dlXComp ) ));

% calculate gradients
totalLoss = dlarray( sum(loss), 'CB' );
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
