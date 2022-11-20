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
        thisEvaluation = ModelEvaluation( "Optimization", setup, ...
                                          verbose = false );
        constraint = -1;
    catch
        constraint = 1;
        obj = NaN;
        return
    end
    
    % set the objective function's output
    switch setup.opt.objective
        case 'ReconLoss'
            obj = thisEvaluation.CVLoss.Validation.Mean.ReconLoss;
        case 'ReconLossSmoothed'
            obj = thisEvaluation.CVLoss.Validation.Mean.ReconLossSmoothed;
        case 'ReconLossRegular'
            obj = thisEvaluation.CVLoss.Validation.Mean.ReconLossRegular;
        case 'AuxModelErrorRate'
            obj = thisEvaluation.CVLoss.Validation.Mean.AuxModelErrorRate;
        case 'ExecutionTime'
            obj = thisEvaluation.CVTiming.Training.Mean.TotalTime;
    end


end

