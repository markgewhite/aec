function [ obj, constraint ] = objectiveFcnAE( hyperparams, setup )
    % Objective function for the autoencoder to be used for optimisation
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
    try
        thisEvaluation = ModelEvaluation( "Optimization", setup, false );
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
        case 'ReconLossRegular'
            obj = thisEvaluation.TestingEvaluation.ReconLossRegular;
        case 'AuxModelLoss'
            obj = thisEvaluation.TestingEvaluation.AuxModelLoss;
    end


end

