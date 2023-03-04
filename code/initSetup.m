function setup = initSetup
    % Specify the configuration where setting differ from default values

    % dataset
    setup.data.class = @JumpGRFDataset;
    setup.data.args.HasNormalizedInput = true;
    setup.data.args.Normalization = 'PAD';
    setup.data.args.ResampleRate = 5;
    setup.data.args.HasAdaptiveTimeSpan = true;
        
    %setup.data.class = @FukuchiDataset;
    %setup.data.args.HasNormalizedInput = true;
    %setup.data.args.FromMatlabFile = false;
    %setup.data.args.HasVGRFOnly = false;
    %setup.data.args.Category = 'JointAngles';
    %setup.data.args.HasPelvis = true;
    %setup.data.args.HasHip = true;
    %setup.data.args.HasKnee = true;

    %setup.data.args.HasVariableLength = true;
    %setup.data.args.TerminationValue = 0.1;

    %setup.data.args.normalization = 'PAD';

    %setup.data.args.normalizedPts = 101;
    %setup.data.args.adaptiveTimeSpan = true;
    %setup.data.args.resampleRate = 1;

    % model
    setup.model.class = @MultiNetFCModel;
    setup.model.args.UsesFdCoefficients = false;
    setup.model.args.ZDim = 3;
    setup.model.args.NumHidden = 1;
    setup.model.args.NumFC = 100;
    setup.model.args.InputDropout = 0;
    setup.model.args.Dropout = 0;
    setup.model.args.ComponentType = 'PDP';
    setup.model.args.AuxModel = 'Logistic';
    setup.model.args.randomSeed = 1234;
    setup.model.args.HasCentredDecoder = true;
    setup.model.args.ShowPlots = true;
    
    % training
    setup.model.args.trainer.NumIterations = 1000;
    setup.model.args.trainer.NumIterPreTrn = 0;
    setup.model.args.trainer.BatchSize = 100;
    setup.model.args.trainer.UpdateFreq = 100;
    setup.model.args.trainer.Holdout = 0;    

    % loss functions
    setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
    setup.model.args.lossFcns.recon.name = 'Reconstruction';
    
    setup.model.args.lossFcns.reconrough.class = @ReconstructionRoughnessLoss;
    setup.model.args.lossFcns.reconrough.name = 'ReconstructionRoughness';
    setup.model.args.lossFcns.reconrough.args.Lambda = 1E-1;
    setup.model.args.lossFcns.reconrough.args.Dilations = [1 2];
    setup.model.args.lossFcns.reconrough.args.UseLoss = false;
    
    %setup.model.args.lossFcns.adv.class = @AdversarialLoss;
    %setup.model.args.lossFcns.adv.name = 'Discriminator';
    %setup.model.args.lossFcns.adv.args.Distribution = 'Cauchy';

    %setup.model.args.lossFcns.kl.class = @KLDivergenceLoss;
    %setup.model.args.lossFcns.kl.name = 'KLDivergence';
    %setup.model.args.lossFcns.kl.args.Beta = 0.1;

    %setup.model.args.lossFcns.fid.class = @FidelityLoss;
    %setup.model.args.lossFcns.fid.name = 'Fidelity';
    %setup.model.args.lossFcns.fid.args.UseLoss = true;

    setup.model.args.lossFcns.zorth.class = @OrthogonalLoss;
    setup.model.args.lossFcns.zorth.name = 'ZOrthogonality';
    setup.model.args.lossFcns.zorth.args.Alpha= 1E0;
    setup.model.args.lossFcns.zorth.args.UseLoss = false;

    setup.model.args.lossFcns.xorth.class = @ComponentLoss;
    setup.model.args.lossFcns.xorth.name = 'XOrthogonality';
    setup.model.args.lossFcns.xorth.args.Criterion = 'Orthogonality';
    setup.model.args.lossFcns.xorth.args.Alpha = 1E1;
    setup.model.args.lossFcns.xorth.args.YLim = [-0.1, 0.5];
    setup.model.args.lossFcns.xorth.args.UseLoss = true;

    setup.model.args.lossFcns.xvar.class = @ComponentLoss;
    setup.model.args.lossFcns.xvar.name = 'XVariance';
    setup.model.args.lossFcns.xvar.args.Criterion = 'Varimax';
    setup.model.args.lossFcns.xvar.args.Alpha = 1E0;
    setup.model.args.lossFcns.xvar.args.YLim = [-0.5, 0];
    setup.model.args.lossFcns.xvar.args.UseLoss = true;
            
    %setup.model.args.lossFcns.cls.class = @ClassifierLoss;
    %setup.model.args.lossFcns.cls.name = 'ZClassifier';
    %setup.model.args.lossFcns.cls.args.UseLoss = false;

    %setup.model.args.lossFcns.xcls.class = @InputClassifierLoss;
    %setup.model.args.lossFcns.xcls.name = 'XClassifier';

    % evaluations
    setup.eval.args.CVType = 'Holdout';
    setup.eval.args.KFolds = 2;
    setup.eval.args.KFoldRepeats = 5;
    setup.eval.args.HasIdenticalPartitions = false;



end