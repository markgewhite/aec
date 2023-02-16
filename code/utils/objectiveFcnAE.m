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
    %try
        thisEvaluation = ModelEvaluation( "Optimization", path, setup );
        constraint = -1;
    %catch
    %    constraint = 1;
    %    obj = NaN;
    %    userdata = [];
    %    return
    %end
    
    % set the objective function's output
    switch setup.opt.objective
        case 'ReconLoss'
            obj = thisEvaluation.CVLoss.Testing.Mean.ReconLoss;
        case 'ReconLossSmoothed'
            obj = thisEvaluation.CVLoss.Testing.Mean.ReconLossSmoothed;
        case 'ReconLossRegular'
            obj = thisEvaluation.CVLoss.Testing.Mean.ReconLossRegular;
        case 'ReconTemporalVarLoss'
            obj = thisEvaluation.CVLoss.Testing.Mean.ReconTemporalVarLoss;  
        case 'ReconVar'
            obj = thisEvaluation.CVLoss.Testing.Mean.ReconVar;
        case 'ReconVarRegular'
            obj = thisEvaluation.CVLoss.Testing.Mean.ReconVarRegular;
        case 'ReconTimeVar'
            obj = mean(thisEvaluation.CVLoss.Testing.Mean.ReconTimeVar);
        case 'ReconTimeVarRegular'
            obj = mean(thisEvaluation.CVLoss.Testing.Mean.ReconTimeVarRegular);
        case 'AuxModelErrorRate'
            obj = thisEvaluation.CVLoss.Testing.Mean.AuxModelErrorRate;
        case 'AuxNetworkErrorRate'
            obj = thisEvaluation.CVLoss.Testing.Mean.AuxNetworkErrorRate;
        case 'ExecutionTime'
            obj = thisEvaluation.CVTiming.Training.Mean.TotalTime;
        case 'LambdaTarget'
            obj = thisEvaluation.Models{1}.FDA.LambdaTarget;
        case 'ReconLoss&AuxModelErrorRate'
            obj = thisEvaluation.CVLoss.Testing.Mean.ReconLoss + ...
                        0.1*thisEvaluation.CVLoss.Testing.Mean.AuxModelErrorRate;
        case 'ReconLoss&AuxModelErrorRateEqual'
            obj = thisEvaluation.CVLoss.Testing.Mean.ReconLoss + ...
                        thisEvaluation.CVLoss.Testing.Mean.AuxModelErrorRate;
        otherwise
            error('Unrecognised objective.');
    end

    % add useful data
    userdata.CurrentIteration = thisEvaluation.Models{1}.Trainer.CurrentIteration;
    userdata.CurrentEpoch = thisEvaluation.Models{1}.Trainer.CurrentEpoch;
    userdata.TrainingLoss = thisEvaluation.CVLoss.Training.Mean;
    userdata.TestingLoss = thisEvaluation.CVLoss.Testing.Mean;
    userdata.AuxModelBeta = thisEvaluation.CVAuxMetrics;
    userdata.LossTrnTrace = thisEvaluation.Models{1}.Trainer.LossTrn;
    userdata.LossValTrace = thisEvaluation.Models{1}.Trainer.LossVal;


end

