function [ obj, constraint, userdata ] = objectiveFcnAE( hyperparams, setup )
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
    
    path = pwd;
    % initialize and run the evaluation
    try
        thisEvaluation = ModelEvaluation( "Optimization", path, setup, ...
                                          verbose = false );
        constraint = -1;
    catch
        constraint = 1;
        obj = -1;
        userdata = [];
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
        case 'ReconVar'
            obj = thisEvaluation.CVLoss.Validation.Mean.ReconVar;
        case 'ReconVarRegular'
            obj = thisEvaluation.CVLoss.Validation.Mean.ReconVarRegular;
        case 'ReconTimeVar'
            obj = mean(thisEvaluation.CVLoss.Validation.Mean.ReconTimeVar);
        case 'ReconTimeVarRegular'
            obj = mean(thisEvaluation.CVLoss.Validation.Mean.ReconTimeVarRegular);
        case 'AuxModelErrorRate'
            obj = thisEvaluation.CVLoss.Validation.Mean.AuxModelErrorRate;
        case 'ExecutionTime'
            obj = thisEvaluation.CVTiming.Training.Mean.TotalTime;
        case 'LambdaTarget'
            obj = thisEvaluation.Models{1}.FDA.LambdaTarget;
        case 'ReconLoss&AuxModelErrorRate'
            obj = thisEvaluation.CVLoss.Validation.Mean.ReconLoss + ...
                        0.1*thisEvaluation.CVLoss.Validation.Mean.AuxModelErrorRate;
        case 'ReconLoss&AuxModelErrorRateEqual'
            obj = thisEvaluation.CVLoss.Validation.Mean.ReconLoss + ...
                        thisEvaluation.CVLoss.Validation.Mean.AuxModelErrorRate;
    end

    % add useful data
    userdata.CurrentIteration = thisEvaluation.Models{1}.Trainer.CurrentIteration;
    userdata.CurrentEpoch = thisEvaluation.Models{1}.Trainer.CurrentEpoch;
    userdata.TrainingLoss = thisEvaluation.CVLoss.Training.Mean;
    userdata.AuxModelBeta = thisEvaluation.CVAuxMetrics;
    userdata.LossTrnTrace = thisEvaluation.Models{1}.Trainer.LossTrn;
    userdata.LossValTrace = thisEvaluation.Models{1}.Trainer.LossVal;
    userdata.MetricsTrace = thisEvaluation.Models{1}.Trainer.Metrics;


end

