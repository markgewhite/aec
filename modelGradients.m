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
w = [ dlnetEnc.Learnables.Value{1}; dlnetDec.Learnables.Value{1}'];

L2Loss = mean( setup.L2Regularization*sum( w.^2 ) );

% --- calculate gradients ---

loss = reconLoss + L2Loss; 

grad.enc = dlgradient( loss, dlnetEnc.Learnables );
grad.dec = dlgradient( loss, dlnetDec.Learnables );
                               
% Calculate the scores
score = 0;

end

