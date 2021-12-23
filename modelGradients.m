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

% --- reconstruction phase ---

% generate latent encodings
[ dlZFake, state.enc ] = forward( dlnetEnc, dlXReal );

% reconstruct curves from latent codes
[ dlXFake, state.dec ] = forward( dlnetDec, dlZFake );

% calculate the reconstruction loss
reconLoss = mean(mean( (dlXFake - dlXReal).^2 ));

% calculate the L2 regularization loss
w = [ dlnetEnc.Learnables.Value{1}; dlnetDec.Learnables.Value{3}'];

L2Loss = setup.weightL2Regularization*mean( sum( w.^2 ) );

% calculate the component roughness loss
% dlXComp = latentComponents( dlnetDec, dlZFake );
% XComp = double(extractdata(dlXComp));
XComp = double(extractdata(dlXFake));
XFd = smooth_basis( setup.fda.tSpan, XComp, setup.fda.fdPar );
XCompD2 = eval_fd( setup.fda.tSpan, XFd, 2 );
dlXCompD2 = dlarray( XCompD2, 'CB' );
lossRoughness = setup.curveD2Regularization*mean( sum( dlXCompD2.^2 ) );
%dlXFakeD2 = dlarray( diff(extractdata(dlXFake),2), 'CB' );
%lossRoughness = setup.curveD2Regularization*mean( sum( dlXFakeD2.^2 ) );


% --- calculate gradients ---

loss = reconLoss + L2Loss + lossRoughness; 
% disp(['lossRoughness = ' num2str(lossRoughness, '%.3f')]);

grad.enc = dlgradient( loss, dlnetEnc.Learnables );
grad.dec = dlgradient( loss, dlnetDec.Learnables );
                               
% Calculate the scores
score = 0;

end

