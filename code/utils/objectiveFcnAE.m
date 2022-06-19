% ************************************************************************
% Function: objectiveFcnAE
%
% Objective function for the autoencoder to be used for optimisation
%
% Parameters:
%           
% Outputs:
%           setup : initialised setup structure
%
% ************************************************************************

function [ obj, constraint ] = objectiveFcnAE( hyperparams, setup )
    arguments
        hyperparams
        setup           struct
    end

% apply the hyperparameters
for i = 1:size( hyperparams, 2 )

    hpName = hyperparams.Properties.VariableNames{i};
    hpValue = hyperparams.(hpName);
    
    hpName = strrep( hpName, "_", "." );
    setup = applySetting( setup, hpName, hpValue );

    setup = updateDependencies( setup, hpName, hpValue );

end

% initialize and run the evaluation
thisEvaluation = ModelEvaluation;
try
    thisEvaluation = thisEvaluation.run( setup, false );
    constraint = -1;
catch
    constraint = 1;
    obj = NaN;
    return
end

% set the objective function's output
switch setup.opt.objective
    case 'ReconLoss'
        obj = thisEvaluation.TestingEvaluation.ReconLoss;
    case 'ReconLossSmoothed'
        obj = thisEvaluation.TestingEvaluation.ReconLossSmoothed;
    case 'AuxLoss'
        obj = thisEvaluation.TestingEvaluation.AuxModeLoss;
end


end

