function setup = initSetup
    % Specify the configuration where setting differ from default values

    % -- data setup --
    setup.data.class = @JumpGRFDataset;
    setup.data.args.HasNormalizedInput = true;
    setup.data.args.Normalization = 'PAD';
    setup.data.args.ResampleRate = 5;
    setup.data.args.HasAdaptiveTimeSpan = true;
    
    setup.data.args.normalizedPts = 21;
        
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
    setup.model.args.trainer.NumIterations = 1;
    setup.model.args.trainer.BatchSize = 100;
    setup.model.args.trainer.UpdateFreq = 5000;
    setup.model.args.trainer.Holdout = 0;
    
    % --- evaluation setup ---
    setup.eval.args.CVType = 'Holdout';
    setup.eval.args.KFolds = 2;
    setup.eval.args.KFoldRepeats = 2;



end