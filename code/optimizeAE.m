% Run optimization searches on the autoencoder

clear;

runAnalysis = false;

% set the destinations for results and figures
path = fileparts( which('code/optimizeAE.m') );
path = [path '/../results/opt/'];

path2 = fileparts( which('code/optimizeAE.m') );
path2 = [path2 '/../paper/results/'];

% -- data setup --
setup.data.class = @SyntheticDataset;
setup.data.args.ClassSizes = [100 100];
setup.data.args.HasNormalizedInput = true;
setup.data.args.NormalizedPts = 51;
zscore = 0.5;

% -- loss functions --
setup.lossFcns.recon.class = @ReconstructionLoss;
setup.lossFcns.recon.name = 'Reconstruction';

% -- model setup --
setup.model.class = @FCModel;
setup.model.args.ZDim = 4;
setup.model.args.InitZDimActive = 0;
setup.model.args.KFolds = 1;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.randomSeed = 1234;
setup.model.args.CompressionLevel = 3;
setup.model.args.ShowPlots = false;

% -- trainer setup --
setup.model.args.trainer.numEpochs = 40; % 400
setup.model.args.trainer.numEpochsPreTrn = 10; %10
setup.model.args.trainer.updateFreq = 200;
setup.model.args.trainer.batchSize = 50;
setup.model.args.trainer.holdout = 0;

% -- optimizer setup --
setup.opt.objective = 'ReconLossRegular';

% define optimizable variables
varDef(1) = optimizableVariable( 'data_args_HasAdaptiveTimeSpan', ...
        ["false" "true"], ...
        'Type', 'categorical', 'Optimize', false );

varDef(2) = optimizableVariable( 'data_args_NormalizedPts', ...
        [3 1000], Type = 'integer', Transform = 'log', ... 
        Optimize = true );

% setup objective function
objFcn = @(x) objectiveFcnAE( x, setup );

% run optimisation
output = bayesopt( objFcn, varDef, ...
            NumCoupledConstraints = 1, ...
            ExplorationRatio = 2.0, ...
            MaxObjectiveEvaluations = 50 );


function [ obj, constraint ] = objWrapper( hyperparams, setup )
    % Objective function wrapper
    arguments
        hyperparams
        setup           struct
    end
    
    switch char(setup.data.class)
        case 'SyntheticDataset'
            setup.data.args.TemplateSeed = randi(10000);
    end

    [obj, constraint] = objectiveFcnAE( hyperparams, setup );

end

