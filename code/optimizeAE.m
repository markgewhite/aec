% Run optimization searches on the autoencoder

clear;

optID = 162;
disp(['OptID = ' num2str(optID)]);

dataset = "Jumps";

% -- optimizer setup --
setup.opt.exploration = 0.5;
setup.opt.numEvaluations = 30;
setup.opt.in_parallel = true;
setup.opt.acquisitionFcnName = 'expected-improvement-plus';

% set the destinations for results and figures
path = fileparts( which('code/optimizeAE.m') );
path = [path '/../results/opt/'];

% -- data setup --
switch dataset
    case "Jumps"
        setup.data.class = @JumpGRFDataset;
        setup.data.args.Normalization = 'PAD';
        setup.data.args.HasNormalizedInput = true;
        setup.data.args.ResampleRate = 10;

    case "UCR"
        setup.data.class = @UCRDataset;
        datasets = [ 17, 31, 38, 74, 104 ];
        setup.data.args.SetID = 85;
        setup.data.args.HasNormalizedInput = true;

    case "Synthetic"
        setup.data.class = @SyntheticDataset;
        setup.data.args.ClassSizes = [200 200];
        setup.data.args.HasNormalizedInput = true;
        zscore = 0.5;
        
        setup.data.args.NumPts = 201;
        setup.data.args.NumTemplatePts = 17;
        setup.data.args.Scaling = [8 4 2 1];
        setup.data.args.Mu = 0.25*[4 3 2 1];
        setup.data.args.Sigma = zscore*setup.data.args.Mu;
        setup.data.args.Eta = 0.1;
        setup.data.args.Tau = 0;    
        setup.data.args.WarpLevel = 1;
        setup.data.args.SharedLevel = 3;

    case 'Synthetic-Legacy'
        setup.data.class = @SyntheticDataset;
        setup.data.args.ClassSizes = [100 100];
        setup.data.args.HasNormalizedInput = true;
        setup.data.args.NormalizedPts = 51;
        zscore = 0.5;

    otherwise
        error("Unrecognised dataset specified.");
    
end

% -- model setup --
setup.model.args.ZDim = 4;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.HasCentredDecoder = true;
setup.model.args.RandomSeed = 1234;
setup.model.args.ShowPlots = false;

% -- loss functions --
setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
setup.model.args.lossFcns.recon.name = 'Reconstruction';

setup.model.args.lossFcns.zcls.class = @ClassifierLoss;
setup.model.args.lossFcns.zcls.name = 'ZClassifier';

setup.model.args.lossFcns.adv.class = @AdversarialLoss;
setup.model.args.lossFcns.adv.name = 'Discriminator';
setup.model.args.lossFcns.adv.args.DoCalcLoss = false;

setup.model.args.lossFcns.kl.class = @KLDivergenceLoss;
setup.model.args.lossFcns.kl.name = 'KLDivergence';
setup.model.args.lossFcns.kl.args.DoCalcLoss = false;

% -- trainer setup --
setup.model.args.trainer.NumIterations = 5000;
setup.model.args.trainer.UpdateFreq = 5000;
setup.model.args.trainer.BatchSize = 5000;
setup.model.args.trainer.Holdout = 0.2;


switch optID

    case 101
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'ReconLossRegular';

        setup.model.args.NumHidden = 1;
        setup.model.args.InputDropout = 0.0;
        setup.model.args.Dropout = 0;

        varDef(1) = optimizableVariable( 'data_args_NormalizedPts', ...
                [3 100], Type = 'integer', Transform = 'log', ... 
                Optimize = true );
        
        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

    case 102
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'AuxModelErrorRate';

        setup.model.args.NumHidden = 1;
        setup.model.args.InputDropout = 0.0;
        setup.model.args.Dropout = 0;

        varDef(1) = optimizableVariable( 'data_args_NormalizedPts', ...
                [3 100], Type = 'integer', Transform = 'log', ... 
                Optimize = true );
        
        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

    case 103
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'LambdaTarget';

        setup.model.args.NumHidden = 1;
        setup.model.args.InputDropout = 0.0;
        setup.model.args.Dropout = 0;

        varDef(1) = optimizableVariable( 'data_args_NormalizedPts', ...
                [3 100], Type = 'integer', Transform = 'log', ... 
                Optimize = true );
        
        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

    case 104
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'ReconVarRegular';

        setup.model.args.NumHidden = 1;
        setup.model.args.InputDropout = 0.0;
        setup.model.args.Dropout = 0;

        varDef(1) = optimizableVariable( 'data_args_NormalizedPts', ...
                [3 100], Type = 'integer', Transform = 'log', ... 
                Optimize = true );
        
        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

    case 105
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'ReconTimeVarRegular';

        setup.model.args.NumHidden = 1;
        setup.model.args.InputDropout = 0.0;
        setup.model.args.Dropout = 0;

        varDef(1) = optimizableVariable( 'data_args_NormalizedPts', ...
                [3 100], Type = 'integer', Transform = 'log', ... 
                Optimize = true );
        
        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

    case 106
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'ReconTimeVar';

        setup.model.args.NumHidden = 1;
        setup.model.args.InputDropout = 0.0;
        setup.model.args.Dropout = 0;

        varDef(1) = optimizableVariable( 'data_args_NormalizedPts', ...
                [3 100], Type = 'integer', Transform = 'log', ... 
                Optimize = true );
        
        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

    case 111
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'ReconLoss';

        setup.model.args.FCFactor = 1;
        setup.model.args.InputDropout = 0.0;
        setup.model.args.Dropout = 0;
        setup.data.args.NormalizedPts = 21;

        varDef(1) = optimizableVariable( 'model_args_NumHidden', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );
        
        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

    case 112
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'AuxModelErrorRate';

        setup.model.args.FCFactor = 1;
        setup.model.args.InputDropout = 0.0;
        setup.model.args.Dropout = 0;
        setup.data.args.NormalizedPts = 21;

        varDef(1) = optimizableVariable( 'model_args_NumHidden', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );
        
        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

    case 113
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'ReconLoss';

        setup.model.args.FCFactor = 1;
        setup.model.args.InputDropout = 0.0;
        setup.model.args.Dropout = 0;
        setup.data.args.NormalizedPts = 101;

        varDef(1) = optimizableVariable( 'model_args_NumHidden', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );
        
        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

    case 121
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'ReconLoss';

        setup.model.args.NumHidden = 2;
        setup.model.args.NumFC = 256;
        setup.model.args.FCFactor = 1;
        setup.model.args.InputDropout = 0.0;
        setup.data.args.NormalizedPts = 21;

        varDef(1) = optimizableVariable( 'model_args_ReLuScale', ...
                [0.001 1.0], Type = 'real', Transform = 'log', ... 
                Optimize = true );
        
        varDef(2) = optimizableVariable( 'model_args_Dropout', ...
                [0.001 0.200], Type = 'real', Transform = 'log', ... 
                Optimize = true );

    case 122
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'AuxModelErrorRate';

        setup.model.args.NumHidden = 2;
        setup.model.args.NumFC = 256;
        setup.model.args.FCFactor = 1;
        setup.model.args.InputDropout = 0.0;
        setup.data.args.NormalizedPts = 21;

        varDef(1) = optimizableVariable( 'model_args_ReLuScale', ...
                [0.001 1.0], Type = 'real', Transform = 'log', ... 
                Optimize = true );
        
        varDef(2) = optimizableVariable( 'model_args_Dropout', ...
                [0.001 0.200], Type = 'real', Transform = 'log', ... 
                Optimize = true );

    case 123
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'ReconLoss';

        setup.model.args.NumHidden = 1;
        setup.model.args.NumFC = 512;
        setup.model.args.FCFactor = 1;
        setup.model.args.InputDropout = 0.0;
        setup.data.args.NormalizedPts = 21;

        varDef(1) = optimizableVariable( 'model_args_ReLuScale', ...
                [0.001 1.0], Type = 'real', Transform = 'log', ... 
                Optimize = true );
        
        varDef(2) = optimizableVariable( 'model_args_Dropout', ...
                [0.001 0.200], Type = 'real', Transform = 'log', ... 
                Optimize = true );

    case 124
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'AuxModelErrorRate';

        setup.model.args.NumHidden = 2;
        setup.model.args.NumFC = 256;
        setup.model.args.FCFactor = 1;
        setup.model.args.InputDropout = 0.0;
        setup.data.args.NormalizedPts = 21;

        varDef(1) = optimizableVariable( 'model_args_ReLuScale', ...
                [0.001 1.0], Type = 'real', Transform = 'log', ... 
                Optimize = true );

    case 131
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'ReconLoss&AuxModelErrorRate';

        setup.model.args.FCFactor = 1;
        setup.model.args.InputDropout = 0.0;

        varDef(1) = optimizableVariable( 'model_args_NumHidden', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );

        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(3) = optimizableVariable( 'model_args_ReLuScale', ...
                [0.001 1.0], Type = 'real', Transform = 'log', ... 
                Optimize = true );

        varDef(4) = optimizableVariable( 'model_args_Dropout', ...
                [0.001 0.200], Type = 'real', Transform = 'log', ... 
                Optimize = true );

    case 132
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a FCModel and a simplified setup
        setup.model.class = @FCModel;
        setup.opt.objective = 'ReconLoss&AuxModelErrorRateEqual';

        setup.model.args.InputDropout = 0.0;
        setup.model.args.ReLuScale = 0.20;
        setup.model.args.Dropout = 0.05;

        varDef(1) = optimizableVariable( 'model_args_NumHidden', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );

        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 2048], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(3) = optimizableVariable( 'model_args_FCFactor', ...
                [1 3], Type = 'integer', ... 
                Optimize = true );


    case 141
        % Find the best combination of output resolution and the number of
        % nodes for an asymmetric FC model
        setup.model.class = @AsymmetricFCModel;
        setup.opt.objective = 'ReconLoss';
        setup.opt.numEvaluations = 50;

        setup.model.args.InputDropout = 0.0;
        setup.model.args.ReLuScale = 0.20;
        setup.model.args.Dropout = 0.05;

        setup.data.args.NormalizedPts = 21;

        varDef(1) = optimizableVariable( 'model_args_NumHidden', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );

        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 2048], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(3) = optimizableVariable( 'model_args_FCFactor', ...
                [1 3], Type = 'integer', ... 
                Optimize = true );

        varDef(4) = optimizableVariable( 'model_args_NumHiddenDecoder', ...
                [1 2], Type = 'integer', ... 
                Optimize = true );

        varDef(5) = optimizableVariable( 'model_args_NumFCDecoder', ...
                [16 2048], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(6) = optimizableVariable( 'model_args_FCFactorDecoder', ...
                [1 3], Type = 'integer', ... 
                Optimize = true );

    case 142
        % Find the best combination of output resolution and the number of
        % nodes for an asymmetric FC model
        setup.model.class = @AsymmetricFCModel;
        setup.opt.objective = 'AuxModelErrorRate';
        setup.opt.numEvaluations = 50;

        setup.model.args.InputDropout = 0.0;
        setup.model.args.ReLuScale = 0.20;
        setup.model.args.Dropout = 0.05;

        setup.data.args.NormalizedPts = 21;

        varDef(1) = optimizableVariable( 'model_args_NumHidden', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );

        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 2048], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(3) = optimizableVariable( 'model_args_FCFactor', ...
                [1 3], Type = 'integer', ... 
                Optimize = true );

        varDef(4) = optimizableVariable( 'model_args_NumHiddenDecoder', ...
                [1 2], Type = 'integer', ... 
                Optimize = true );

        varDef(5) = optimizableVariable( 'model_args_NumFCDecoder', ...
                [16 2048], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(6) = optimizableVariable( 'model_args_FCFactorDecoder', ...
                [1 3], Type = 'integer', ... 
                Optimize = true );

    case 143
        % Find the best combination of output resolution and the number of
        % nodes for an asymmetric FC model
        setup.model.class = @AsymmetricFCModel;
        setup.opt.objective = 'ReconLoss&AuxModelErrorRateEqual';
        setup.opt.numEvaluations = 50;

        setup.model.args.InputDropout = 0.0;
        setup.model.args.ReLuScale = 0.20;
        setup.model.args.Dropout = 0.05;

        setup.data.args.NormalizedPts = 21;

        varDef(1) = optimizableVariable( 'model_args_NumHidden', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );

        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 2048], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(3) = optimizableVariable( 'model_args_FCFactor', ...
                [1 3], Type = 'integer', ... 
                Optimize = true );

        varDef(4) = optimizableVariable( 'model_args_NumHiddenDecoder', ...
                [1 2], Type = 'integer', ... 
                Optimize = true );

        varDef(5) = optimizableVariable( 'model_args_NumFCDecoder', ...
                [16 2048], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(6) = optimizableVariable( 'model_args_FCFactorDecoder', ...
                [1 3], Type = 'integer', ... 
                Optimize = true );

   case 151
        % Find the best combination of output resolution and the number of
        % nodes for an asymmetric FC model
        setup.model.class = @AsymmetricFCModel;
        setup.opt.objective = 'ReconLoss';
        setup.opt.numEvaluations = 40;

        setup.model.args.FCFactor = 1;
        setup.model.args.FCFactorDecoder = 1;
        
        setup.model.args.InputDropout = 0.0;
        setup.model.args.ReLuScale = 0.20;
        setup.model.args.Dropout = 0.05;

        setup.data.args.NormalizedPts = 21;

        varDef(1) = optimizableVariable( 'model_args_NumHidden', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );

        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 2048], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(3) = optimizableVariable( 'model_args_NumHiddenDecoder', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );

        varDef(4) = optimizableVariable( 'model_args_NumFCDecoder', ...
                [16 2048], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

   case 161 
        % Find the best combination of output resolution and the number of
        % nodes for an asymmetric FC model
        setup.model.class = @AsymmetricFCModel;
        setup.opt.objective = 'ReconLoss';
        setup.opt.numEvaluations = 80;

        setup.model.args.FCFactor = 2;
        setup.model.args.FCFactorDecoder = 2;
        
        setup.model.args.InputDropout = 0.0;
        setup.model.args.ReLuScale = 0.20;
        setup.model.args.Dropout = 0.05;

        setup.data.args.NormalizedPts = 21;

        varDef(1) = optimizableVariable( 'model_args_NumHidden', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );

        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 2048], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(3) = optimizableVariable( 'model_args_NumHiddenDecoder', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );

        varDef(4) = optimizableVariable( 'model_args_NumFCDecoder', ...
                [16 2048], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

   case 162
        % Find the best combination of output resolution and the number of
        % nodes for an asymmetric FC model, considering the auxiliary
        % network
        setup.model.class = @AsymmetricFCModel;
        setup.opt.objective = 'AuxNetworkErrorRate';
        setup.opt.numEvaluations = 80;

        setup.model.args.FCFactor = 2;
        setup.model.args.FCFactorDecoder = 2;
        
        setup.model.args.InputDropout = 0.0;
        setup.model.args.ReLuScale = 0.20;
        setup.model.args.Dropout = 0.05;

        setup.data.args.NormalizedPts = 25;

        varDef(1) = optimizableVariable( 'model_args_NumHidden', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );

        varDef(2) = optimizableVariable( 'model_args_NumFC', ...
                [16 2048], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(3) = optimizableVariable( 'model_args_NumHiddenDecoder', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );

        varDef(4) = optimizableVariable( 'model_args_NumFCDecoder', ...
                [16 2048], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(5) = optimizableVariable( 'model_args_lossFcns_zcls_NumHidden', ...
                [1 5], Type = 'integer', ... 
                Optimize = true );

        varDef(6) = optimizableVariable( 'model_args_lossFcns_zcls_NumFC', ...
                [8 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );



   case 171 
        % Find a good demonstration of a simple model
        setup.model.class = @FCModel;
        setup.opt.objective = 'ReconLossRegular';
        setup.opt.numEvaluations = 20;

        setup.model.args.FCFactor = 1;       
        setup.model.args.ReLuScale = 1;
        setup.model.args.InputDropout = 0;
        setup.model.args.Dropout = 0;
        setup.model.args.NetNormalizationType = 'Layer';

        varDef(1) = optimizableVariable( 'model_args_NumFC', ...
                [4 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(2) = optimizableVariable( 'data_args_NormalizedPts', ...
                [5 100], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

   case 172
        % Find a good demonstration of a simple model
        setup.model.class = @FCModel;
        setup.opt.objective = 'AuxModelErrorRate';
        setup.opt.numEvaluations = 20;

        setup.model.args.FCFactor = 1;       
        setup.model.args.ReLuScale = 1;
        setup.model.args.InputDropout = 0;
        setup.model.args.Dropout = 0;
        setup.model.args.NetNormalizationType = 'Layer';

        varDef(1) = optimizableVariable( 'model_args_NumFC', ...
                [4 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(2) = optimizableVariable( 'data_args_NormalizedPts', ...
                [5 100], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

   case 173
        % Find a good demonstration of a simple model
        setup.model.class = @FCModel;
        setup.opt.objective = 'ReconLoss&AuxModelErrorRateEqual';
        setup.opt.numEvaluations = 20;

        setup.model.args.FCFactor = 1;       
        setup.model.args.ReLuScale = 1;
        setup.model.args.InputDropout = 0;
        setup.model.args.Dropout = 0;
        setup.model.args.NetNormalizationType = 'Layer';

        varDef(1) = optimizableVariable( 'model_args_NumFC', ...
                [4 512], Type = 'integer', Transform = 'log', ... 
                Optimize = true );

        varDef(2) = optimizableVariable( 'data_args_NormalizedPts', ...
                [5 100], Type = 'integer', Transform = 'log', ... 
                Optimize = true );



    case 201
        % Find the best combination of output resolution and the number of
        % nodes with one hidden layer for a ConvModel and a simplified setup
        setup.model.class = @ConvolutionalModel;
        setup.model.args.HasFCDecoder = true;
        setup.opt.objective = 'AuxModelErrorRate';

        seteup.data.args.ResampleRate = 5;
        setup.model.args.NumFilters= 16;

        setup.model.args.trainer.NumIterations = 1000;

        setup.model.args.NumHidden = 1;
        setup.model.args.NumFC = 256;
        setup.model.args.FCFactor = 1;
        setup.model.args.InputDropout = 0.0;
        setup.model.args.ReLuScale = 0.0;
        setup.model.args.Dropout = 0.0;

        varDef(1) = optimizableVariable( 'model_args_FilterSize', ...
                [5 13], Type = 'integer', ... 
                Optimize = true );

        numEvaluations = 15;
        
    otherwise
        error('Unrecognised optID.');

end


% setup objective function
objFcn = @(x) objectiveFcnAE( x, setup );
objWrapperFcn = @(x) objWrapper( x, setup );

% run optimisation
output = bayesopt( objWrapperFcn, varDef, ...
            NumCoupledConstraints = 1, ...
            ExplorationRatio = setup.opt.exploration, ...
            MaxObjectiveEvaluations = setup.opt.numEvaluations, ...
            AcquisitionFunctionName = setup.opt.acquisitionFcnName, ...
            UseParallel = setup.opt.in_parallel);

% save optimisation data
filename = ['Bayesopt-' num2str(optID)];
save( fullfile(path, filename), 'output', 'setup' );

function [ obj, constraint, userdata ] = objWrapper( hyperparams, setup )
    % Objective function wrapper
    arguments
        hyperparams
        setup           struct
    end
    
    switch char(setup.data.class)
        case 'SyntheticDataset'
            setup.data.args.TemplateSeed = randi(10000);
    end

    [obj, constraint, userdata] = objectiveFcnAE( hyperparams, setup );

end

