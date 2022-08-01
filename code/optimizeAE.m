% Run optimization searches on the autoencoder

clear;

dataset = "Jumps";

exploration = 0.5;

% set the destinations for results and figures
path = fileparts( which('code/optimizeAE.m') );
path = [path '/../results/opt/'];

path2 = fileparts( which('code/optimizeAE.m') );
path2 = [path2 '/../paper/results/'];

% -- data setup --
switch dataset
    case "Jumps"
        setup.data.class = @JumpGRFDataset;
        setup.data.args.Normalization = 'PAD';
        setup.data.args.HasNormalizedInput = false;
        setup.data.args.ResampleRate = 10;

    case "Synthetic"
        setup.data.class = @SyntheticDataset;
        setup.data.args.ClassSizes = [200 200];
        setup.data.args.HasNormalizedInput = false;
        zscore = 0.5;
        
        setup.data.args.NumPts = 201;
        setup.data.args.NumTemplatePts = 17;
        setup.data.args.Scaling = [8 4 2 1];
        setup.data.args.Mu = 0.25*[4 3 2 1];
        setup.data.args.Sigma = zscore*setup.data.args.Mu;
        setup.data.args.Eta = 0.1;
        setup.data.args.Tau = 0;    
        setup.data.args.WarpLevel = 1;
        setup.data.args.SharedLevel = 2;

    otherwise
        error("Unrecognised dataset specified.");
    
end

% -- loss functions --
setup.lossFcns.recon.class = @ReconstructionLoss;
setup.lossFcns.recon.name = 'Reconstruction';
setup.lossFcns.zcls.class = @ClassifierLoss;
setup.lossFcns.zcls.name = 'ZClassifier';

% -- model setup --
setup.model.class = @TCNModel;
setup.model.args.ZDim = 4;
setup.model.args.InitZDimActive = 0;
setup.model.args.KFolds = 1;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.randomSeed = 1234;
setup.model.args.CompressionLevel = 3;
setup.model.args.ShowPlots = false;
setup.model.args.HasFCDecoder = true;
setup.model.args.FCFactor = 1;
setup.model.args.NumFC = 128;
setup.model.args.NumConvHidden = 5;
setup.model.args.DilationFactor = 1;

% -- trainer setup --
setup.model.args.trainer.numEpochs = 100; % 400
setup.model.args.trainer.numEpochsPreTrn = 10; %10
setup.model.args.trainer.updateFreq = 200;
setup.model.args.trainer.batchSize = 50;
setup.model.args.trainer.holdout = 0;

% -- optimizer setup --
setup.opt.objective = 'AuxModelLoss';

% define optimizable variables
varDef(1) = optimizableVariable( 'data_args_HasAdaptiveTimeSpan', ...
        ["false" "true"], Type = 'categorical', ...
        Optimize = false );

varDef(2) = optimizableVariable( 'data_args_NormalizedPts', ...
        [3 1000], Type = 'integer', Transform = 'log', ... 
        Optimize = false );

varDef(3) = optimizableVariable( 'data_args_ResampleRate', ...
        [1 100], Type = 'real', Transform = 'log', ... 
        Optimize = false );

varDef(4) = optimizableVariable( 'data_args_Lambda', ...
        [1E-10 1E10], Type = 'real', Transform = 'log', ... 
        Optimize = false );

varDef(5) = optimizableVariable( 'model_args_HasInputNormalization', ...
        ["false" "true"], Type = 'categorical', ...
        Optimize = false );

% FC Model hyperparameters
varDef(6) = optimizableVariable( 'model_args_NumHidden', ...
        [1 3], Type = 'integer', ... 
        Optimize = false );

varDef(7) = optimizableVariable( 'model_args_NumFC', ...
        [16 256], Type = 'integer', Transform = 'log', ... 
        Optimize = false );

varDef(8) = optimizableVariable( 'model_args_FCFactor', ...
        [1 3], Type = 'integer', ... 
        Optimize = false );

varDef(9) = optimizableVariable( 'model_args_ReLuScale', ...
        [0.01 0.9], Type = 'real', Transform = 'log', ... 
        Optimize = false );

varDef(10) = optimizableVariable( 'model_args_InputDropout', ...
        [0.01 0.5], Type = 'real', Transform = 'log', ... 
        Optimize = false );

varDef(11) = optimizableVariable( 'model_args_Dropout', ...
        [0.01 0.5], Type = 'real', Transform = 'log', ... 
        Optimize = false );


% data hyperparameters
varDef(12) = optimizableVariable( 'data_args_NumPts', ...
        [20 2000], Type = 'integer', Transform = 'log', ... 
        Optimize = false );

% TCN Model hyperparameters
varDef(13) = optimizableVariable( 'model_args_HasReluInside', ...
        ["false" "true"], Type = 'categorical', ...
        Optimize = false );

varDef(14) = optimizableVariable( 'model_args_NumConvHidden', ...
        [2 8], Type = 'integer', ... 
        Optimize = false );

varDef(15) = optimizableVariable( 'model_args_DilationFactor', ...
        [1 2], Type = 'integer', ... 
        Optimize = false );

varDef(16) = optimizableVariable( 'model_args_FilterSize', ...
        [3 11], Type = 'integer', ... 
        Optimize = true );

varDef(17) = optimizableVariable( 'model_args_NumFilters', ...
        [4 64], Type = 'integer', Transform = 'log', ... 
        Optimize = true );



% setup objective function
objFcn = @(x) objectiveFcnAE( x, setup );

% run optimisation
output = bayesopt( objFcn, varDef, ...
            NumCoupledConstraints = 1, ...
            ExplorationRatio = exploration, ...
            MaxObjectiveEvaluations = 30 );


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

