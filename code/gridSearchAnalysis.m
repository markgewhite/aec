% run a grid search

name = "DatasetA";

% set the destinations for results and figures
path0 = fileparts( which('code/gridSearchAnalysis.m') );
path = [path0 '/../results/grid/'];
pathResults = [path0 '/../paper/results/'];

% -- data setup --
setup.data.args.HasNormalizedInput = true;
setup.data.args.normalizedPts = 21;

% -- model setup --
setup.model.class = @BranchedFCModel;
setup.model.args.NumHidden = 1;
setup.model.args.NumFC = 20;
setup.model.args.InputDropout = 0;
setup.model.args.Dropout = 0;
setup.model.args.NetNormalizationType = 'None';
setup.model.args.NetActivationType = 'None';

setup.model.args.NumHiddenDecoder = 2;
setup.model.args.NumFCDecoder = 10;
setup.model.args.FCFactorDecoder = 0;
setup.model.args.NetNormalizationTypeDecoder = 'None';
setup.model.args.NetActivationTypeDecoder = 'None';

setup.model.args.ComponentType = 'PDP';
setup.model.args.AuxModel = 'Logistic';
setup.model.args.randomSeed = 1234;
setup.model.args.HasCentredDecoder = true;
setup.model.args.ShowPlots = true;

% -- loss functions --
setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
setup.model.args.lossFcns.recon.name = 'Reconstruction';

setup.model.args.lossFcns.reconrough.class = @ReconstructionRoughnessLoss;
setup.model.args.lossFcns.reconrough.name = 'ReconstructionRoughness';

setup.model.args.lossFcns.zorth.class = @OrthogonalLoss;
setup.model.args.lossFcns.zorth.name = 'ZOrthogonality';

setup.model.args.lossFcns.xvar.class = @ComponentLoss;
setup.model.args.lossFcns.xvar.name = 'XVarimax';
setup.model.args.lossFcns.xvar.args.Criterion = 'Varimax';

setup.model.args.lossFcns.zcls.class = @ClassifierLoss;
setup.model.args.lossFcns.zcls.name = 'ZClassifier';
setup.model.args.lossFcns.zcls.args.NumHidden = 1;
setup.model.args.lossFcns.zcls.args.NumFC= 10;
setup.model.args.lossFcns.zcls.args.HasBatchNormalization = false;
setup.model.args.lossFcns.zcls.args.ReluScale = 0;
setup.model.args.lossFcns.zcls.args.Dropout = 0;

% -- trainer setup --
setup.model.args.trainer.NumIterations = 1000;
setup.model.args.trainer.BatchSize = 100;
setup.model.args.trainer.UpdateFreq = 5000;
setup.model.args.trainer.Holdout = 0;

% --- evaluation setup ---
setup.eval.args.CVType = 'Holdout';
setup.eval.args.KFolds = 2;
setup.eval.args.KFoldRepeats = 2;

% grid search
parameters = [ "model.args.ZDim", ...
               "model.args.NumFC", ...
               "model.args.NumFCDecoder", ...
               "model.args.NetNormalizationType", ...
               "model.args.NetActivationType", ...
               "model.args.NetNormalizationTypeDecoder", ...
               "model.args.NetActivationTypeDecoder", ...
               ];
values = {[2 5], ...
          [10 50], ...
          [10 50], ...
          {'None', 'Batch'}, ...
          {'None', 'Relu', 'Tanh'}, ...
          {'None', 'Batch', 'Layer'}, ...
          {'None', 'Relu', 'Tanh'}, ...
          };


switch name

    case "JumpsVGRF"
        % JumpsVGRF data set
        setup.data.class = @JumpGRFDataset;
        setup.data.args.Normalization = 'PAD';
        setup.data.args.ResampleRate = 5;

    case "DatasetA"
        % Data set A: two double-Gaussian classes
        setup.data.class = @ExemplarDataset;   
        setup.data.args.FeatureType = 'Gaussian';
        N = 400;
        sigma = 0.5;
        setup.data.args.ClassSizes = [ N/2 N/2 ];
        setup.data.args.ClassElements = 2;
        setup.data.args.ClassPeaks = [ 2.50 0.0; 2.75 1.0 ];
        setup.data.args.ClassPositions = [ -1 1.5; -1 2 ];
        setup.data.args.ClassWidths = [ 3.0 0.50; 2.5 1.0 ];

        setup.data.args.Covariance{1} = 0.1*[1 0 -sigma; 
                                             0 1 0;
                                             -sigma 0 1];

        setup.data.args.Covariance{2} = 0.05*[1 0 -sigma; 
                                             0 1 0;
                                             -sigma 0 1];

    case "Fukuchi-GRF-3D"
        % Fukuchi data set
        setup.data.class = @FukuchiDataset;
        setup.data.args.YReference = 'AgeGroup';
        setup.data.args.Category = 'Ground';
        setup.data.args.NormalizedPts = 5;
        setup.model.args.trainer.BatchSize = 1000;
        setup.model.args.trainer.InParallel = true;
        setup.model.args.trainer.DoUseGPU = true;


end

myInvestigation = ParallelInvestigation( name, path, parameters, values, setup );

myInvestigation.run;

myInvestigation.saveReport;

myInvestigation.save;



