function setup = initSetup
    % Specify the configuration where setting differ from default values

    % dataset
    setup.data.class = @UCRDataset;
    setup.data.args.SetID = 85;

    %setup.data.class = @JumpGRFDataset;
    %setup.data.args.Normalization = 'PAD';
    %setup.data.args.HasNormalizedInput = true;
    %setup.data.args.NormalizedPts = 51;
    %setup.data.args.HasAdaptiveTimeSpan = true;
    %setup.data.args.ResampleRate = 10;
    %setup.data.args.HasMatchingOutput = false;
        
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
    setup.model.class = @FCModel;
    %setup.model.args.HasFCDecoder = false;
    setup.model.args.ZDim = 4;
    setup.model.args.InitZDimActive = 1;
    setup.model.args.AuxModel = 'Logistic';
    setup.model.args.randomSeed = 1234;
    setup.model.args.HasCentredDecoder = true;
    setup.model.args.ShowPlots = true;
    
    % training
    setup.model.args.trainer.updateFreq = 1;
    setup.model.args.trainer.valType = 'Reconstruction';
    setup.model.args.trainer.numEpochs = 1;
    setup.model.args.trainer.numEpochsPreTrn = 0;
    setup.model.args.trainer.activeZFreq = 10;
    setup.model.args.trainer.batchSize = 40;
    setup.model.args.trainer.holdout = 0;

    % loss functions
    setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
    setup.model.args.lossFcns.recon.name = 'Reconstruction';

    %setup.model.args.lossFcns.adv.class = @AdversarialLoss;
    %setup.model.args.lossFcns.adv.name = 'Discriminator';
    %setup.model.args.lossFcns.adv.args.Distribution = 'DoubleGaussian';
    
    setup.model.args.lossFcns.cls.class = @ClassifierLoss;
    setup.model.args.lossFcns.cls.name = 'ZClassifier';

    %setup.model.args.lossFcns.xcls.class = @InputClassifierLoss;
    %setup.model.args.lossFcns.xcls.name = 'XClassifier';

    % evaluations
    setup.eval.args.verbose = true;
    setup.eval.args.CVType = 'KFold';
    setup.eval.args.KFolds = 10;
    setup.eval.args.HasIdenticalPartitions = true;



end