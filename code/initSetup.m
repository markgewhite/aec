function setup = initSetup
    % Specify the configuration where setting differ from default values

    % dataset
    setup.data.class = @JumpGRFDataset;
    setup.data.args.Normalization = 'PAD';
    setup.data.args.HasNormalizedInput = true;
    setup.data.args.ResampleRate = 5;
        
    %setup.data.class = @FukuchiDataset;
    %setup.data.args.FromMatlabFile = true;
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
    setup.model.class = @AsymmetricFCModel;
    setup.model.args.ZDim = 2;
    setup.model.args.NumFCDecoder = 10;
    setup.model.args.AuxModel = 'Logistic';
    setup.model.args.randomSeed = 1234;
    setup.model.args.HasCentredDecoder = true;
    setup.model.args.ShowPlots = true;
    
    % training
    setup.model.args.trainer.NumIterations = 2000;
    setup.model.args.trainer.NumIterPreTrn = 0;
    setup.model.args.trainer.BatchSize = 200;
    setup.model.args.trainer.UpdateFreq = 100;
    setup.model.args.trainer.Holdout = 0;    

    % loss functions
    setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
    setup.model.args.lossFcns.recon.name = 'Reconstruction';

    setup.model.args.lossFcns.reconvar.class = @ReconstructionTemporalVarLoss;
    setup.model.args.lossFcns.reconvar.name = 'ReconstructionTemporalVariance';
    setup.model.args.lossFcns.reconvar.args.Beta = 1E1;
    setup.model.args.lossFcns.reconvar.args.Gamma = 0;
    
    %setup.model.args.lossFcns.adv.class = @AdversarialLoss;
    %setup.model.args.lossFcns.adv.name = 'Discriminator';
    %setup.model.args.lossFcns.adv.args.distribution = 'Cauchy';

    %setup.model.args.lossFcns.kl.class = @KLDivergenceLoss;
    %setup.model.args.lossFcns.kl.name = 'KLDivergence';
    %setup.model.args.lossFcns.kl.args.beta = 0.1;

    %setup.model.args.lossFcns.fid.class = @FidelityLoss;
    %setup.model.args.lossFcns.fid.name = 'Fidelity';
    %setup.model.args.lossFcns.fid.args.useLoss = true;

    %setup.model.args.lossFcns.zorth.class = @OrthogonalLoss;
    %setup.model.args.lossFcns.zorth.name = 'ZOrthogonality';
    %setup.model.args.lossFcns.zorth.args.useLoss = true;

    %setup.model.args.lossFcns.xorth.class = @ComponentLoss;
    %setup.model.args.lossFcns.xorth.name = 'XOrthogonality';
    %setup.model.args.lossFcns.xorth.args.useLoss = true;
            
    setup.model.args.lossFcns.cls.class = @ClassifierLoss;
    setup.model.args.lossFcns.cls.name = 'ZClassifier';
    setup.model.args.lossFcns.cls.args.useLoss = true;

    %setup.model.args.lossFcns.xcls.class = @InputClassifierLoss;
    %setup.model.args.lossFcns.xcls.name = 'XClassifier';

    % evaluations
    setup.eval.args.CVType = 'Holdout';
    setup.eval.args.KFolds = 4;
    setup.eval.args.KFoldRepeats = 1;
    setup.eval.args.HasIdenticalPartitions = true;



end