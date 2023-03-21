function setup = initSetup
    % Specify the configuration where setting differ from default values

    % dataset
    %setup.data.class = @JumpGRFDataset;
    %setup.data.args.HasNormalizedInput = true;
    %setup.data.args.Normalization = 'PAD';
    %setup.data.args.ResampleRate = 5;
    %setup.data.args.HasAdaptiveTimeSpan = true;
    
    setup.data.args.normalizedPts = 21;
        
    setup.data.class = @FukuchiDataset;
    setup.data.args.HasNormalizedInput = true;
    setup.data.args.FromMatlabFile = false;
    setup.data.args.HasVGRFOnly = false;
    setup.data.args.Category = 'JointAngles';
    setup.data.args.HasPelvis = true;
    setup.data.args.HasHip = true;
    setup.data.args.HasKnee = true;

    %setup.data.args.HasVariableLength = true;
    %setup.data.args.TerminationValue = 0.1;

    % model
    setup.model.class = @BranchedFCModel;
    setup.model.args.UsesFdCoefficients = false;
    setup.model.args.ZDim = 3;
    setup.model.args.ZDimAux = 3;
    setup.model.args.NumFC = 100;
    setup.model.args.NumHiddenDecoder = 1;
    setup.model.args.NumFCDecoder = 10;
    setup.model.args.FCFactorDecoder = 0;
    setup.model.args.InputDropout = 0;
    setup.model.args.Dropout = 0;
    setup.model.args.NetNormalizationType = 'Batch';
    setup.model.args.ComponentType = 'PDP';
    setup.model.args.AuxModel = 'Logistic';
    setup.model.args.randomSeed = 1234;
    %setup.model.args.HasBranchedEncoder = false;
    %setup.model.args.HasEncoderMasking = false;
    setup.model.args.HasBranchedDecoder = true;
    setup.model.args.HasDecoderMasking = true;
    setup.model.args.HasCentredDecoder = true;
    setup.model.args.ShowPlots = true;
    
    % training
    setup.model.args.trainer.NumIterations = 500;
    setup.model.args.trainer.NumIterPreTrn = 0;
    setup.model.args.trainer.BatchSize = 164;
    setup.model.args.trainer.UpdateFreq = 250;
    setup.model.args.trainer.Holdout = 0;    

    % loss functions
    setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
    setup.model.args.lossFcns.recon.name = 'Reconstruction';
    
    setup.model.args.lossFcns.reconrough.class = @ReconstructionRoughnessLoss;
    setup.model.args.lossFcns.reconrough.name = 'ReconstructionRoughness';
    setup.model.args.lossFcns.reconrough.args.UseLoss = true;

    setup.model.args.lossFcns.zorth.class = @OrthogonalLoss;
    setup.model.args.lossFcns.zorth.name = 'ZOrthogonality';
    setup.model.args.lossFcns.zorth.args.Alpha= 1E0;
    setup.model.args.lossFcns.zorth.args.UseLoss = true;

    setup.model.args.lossFcns.xorth.class = @ComponentLoss;
    setup.model.args.lossFcns.xorth.name = 'XOrthogonality';
    setup.model.args.lossFcns.xorth.args.Criterion = 'Orthogonality';
    setup.model.args.lossFcns.xorth.args.Alpha = 1E0;
    setup.model.args.lossFcns.xorth.args.YLim = [0, 0.2];
    setup.model.args.lossFcns.xorth.args.UseLoss = false;

    setup.model.args.lossFcns.xvar.class = @ComponentLoss;
    setup.model.args.lossFcns.xvar.name = 'XVariance';
    setup.model.args.lossFcns.xvar.args.Criterion = 'Varimax2';
    setup.model.args.lossFcns.xvar.args.Alpha = 1E1;
    setup.model.args.lossFcns.xvar.args.YLim = [0, 0.25];
    setup.model.args.lossFcns.xvar.args.UseLoss = true;
            
    %setup.model.args.lossFcns.cls.class = @ClassifierLoss;
    %setup.model.args.lossFcns.cls.name = 'ZClassifier';
    %setup.model.args.lossFcns.cls.args.UseLoss = true;

    %setup.model.args.lossFcns.xcls.class = @InputClassifierLoss;
    %setup.model.args.lossFcns.xcls.name = 'XClassifier';

    % evaluations
    setup.eval.args.CVType = 'KFold';
    setup.eval.args.KFolds = 2;
    setup.eval.args.KFoldRepeats = 2;
    setup.eval.args.HasIdenticalPartitions = false;
    setup.eval.args.InParallel = false;



end