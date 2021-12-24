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

XCompD1 = diff( XComp, 1 );
tp = XCompD1(1:end-1,:).*XCompD1(2:end,:);
nTP = sum( tp<0 );
loss(4) = setup.tpRegularization*mean( nTP );

%dlZReal = dlarray( randn( setup.zDim, setup.batchSize ), 'CB' ); 
%dlXComp = forward( dlnetDec, dlZReal );
%XComp = double(extractdata(dlXComp));
%XFd = smooth_basis( setup.fda.tSpan, XComp, setup.fda.fdPar );
%XCompD2 = eval_fd( setup.fda.tSpan, XFd, 2 );
%XCompD2 = diff( XComp, 2 );
%lossRoughness = setup.curveD2Regularization*mean( sum( XCompD2.^2 ) );

%XFd = smooth_basis( setup.fda.tSpan, XComp, setup.fda.fdPar );
%XCompD2 = eval_fd( setup.fda.tSpan, XFd, 2 );
%dlXCompD2 = dlarray( XCompD2, 'CB' );
%lossRoughness = setup.curveD2Regularization*mean( sum( dlXCompD2.^2 ) );
%dlXFakeD2 = dlarray( diff(extractdata(dlXFake),2), 'CB' );


% --- calculate gradients ---
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


