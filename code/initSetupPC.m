function setup = initSetupPC
    % Specify the configuration where setting differ from default values

    % dataset
    setup.data.class = @JumpGRFDataset;
    setup.data.args.Normalization = 'PAD';
    setup.data.args.HasNormalizedInput = true;
    setup.data.args.NormalizedPts = 51;
    setup.data.args.HasAdaptiveTimeSpan = true;
    setup.data.args.ResampleRate = 10;
    setup.data.args.OverSmoothing = 1E3;
    setup.data.args.HasMatchingOutput = false;
        
    %setup.data.class = @fukuchiDataset;
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

    % loss functions
    setup.lossFcns.recon.class = @ReconstructionLoss;
    setup.lossFcns.recon.name = 'Reconstruction';

    setup.lossFcns.adv.class = @AdversarialLoss;
    setup.lossFcns.adv.name = 'Discriminator';
    
    setup.lossFcns.orth.class = @ComponentLoss;
    setup.lossFcns.orth.name = 'XOrthogonality';
    setup.lossFcns.orth.args.criterion = 'InnerProduct';

    %setup.lossFcns.cls.class = @ClassifierLoss;
    %setup.lossFcns.cls.name = 'Classification';
    %setup.lossFcns.cls.args.useLoss = true;

    %setup.lossFcns.xcls.class = @InputClassifierLoss;
    %setup.lossFcns.xcls.name = 'XClassification';
    %setup.lossFcns.xcls.args.useLoss = true;

    % model
    setup.model.class = @FCModel;
    %setup.model.args.HasFCDecoder = false;
    setup.model.args.ZDim = 4;
    setup.model.args.InitZDimActive = 0;
    setup.model.args.KFolds = 10;
    setup.model.args.IdenticalPartitions = true;
    setup.model.args.AuxModel = 'Logistic';
    
    % training
    setup.model.args.trainer.updateFreq = 5;
    setup.model.args.trainer.valType = 'AuxModel';
    setup.model.args.trainer.numEpochs = 5;
    setup.model.args.trainer.numEpochsPreTrn = 2;
    setup.model.args.trainer.activeZFreq = 10;
    setup.model.args.trainer.batchSize = 40;
    setup.model.args.trainer.holdout = 0;

    % reproducibility settings
    setup.model.args.randomSeed = 1234;
    setup.model.args.randomSeedResets = true;
    setup.model.args.identicalNetInit = true;
    setup.model.args.trainer.miniBatchShuffle = false;


end