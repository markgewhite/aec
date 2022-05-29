function setup = initSetup
    % Specify the configuration where setting differ from default values

    % dataset
    setup.data.class = @jumpGRFDataset;
    setup.data.args.normalization = 'PAD';
    setup.data.args.normalizeInput = false;
    setup.data.args.normalizedPts = 51;
    setup.data.args.hasAdaptiveTimeSpan = true;
    setup.data.args.resampleRate = 10;
        
    %setup.data.class = @fukuchiDataset;
    %setup.data.args.FromMatlabFile = true;
    %setup.data.args.HasVGRFOnly = false;
    %setup.data.args.Category = 'JointAngles';
    %setup.data.args.HasPelvis = true;
    %setup.data.args.HasHip = true;
    %setup.data.args.HasKnee = true;
    %setup.data.args.overSmoothing = 1E5;

    %setup.data.class = @exemplarDataset;   
    %setup.data.args.ClassSizes = 500;
    
    %setup.data.args.ClassMeans = [-1 1];
    %setup.data.args.ClassSDs = [0.5 1.0];
    %setup.data.args.ClassPeaks = [5.0 10];
    %setup.data.args.MeanCovariance{1} = [0.2 0.2; 0.2 0.3];
    %setup.data.args.SDCovariance{1} = [1 0.5; 0.5 1];
    %setup.data.args.PeakCovariance{1} = [1 -0.5; -0.5 1];

    %setup.data.args.HasVariableLength = true;
    %setup.data.args.TerminationValue = 0.1;

    %setup.data.args.normalization = 'PAD';

    %setup.data.args.normalizedPts = 101;
    %setup.data.args.adaptiveTimeSpan = true;
    %setup.data.args.resampleRate = 1;

    % loss functions
    setup.lossFcns.recon.class = @reconstructionLoss;
    setup.lossFcns.recon.name = 'Reconstruction';

    setup.lossFcns.adv.class = @adversarialLoss;
    setup.lossFcns.adv.name = 'Discriminator';

    %setup.lossFcns.mmd.class = @wassersteinLoss;
    %setup.lossFcns.mmd.name = 'MMDDiscriminator';
    %setup.lossFcns.mmd.args.kernel = 'IMQ';
    %setup.lossFcns.mmd.args.useLoss = false;

    setup.lossFcns.orth.class = @componentLoss;
    setup.lossFcns.orth.name = 'Orthogonality';
    setup.lossFcns.orth.args.nSample = 100;
    setup.lossFcns.orth.args.criterion = 'Orthogonality';

    setup.lossFcns.var.class = @componentLoss;
    setup.lossFcns.var.name = 'ExplainedVariance';
    setup.lossFcns.var.args.nSample = 100;
    setup.lossFcns.var.args.criterion = 'ExplainedVariance';
    setup.lossFcns.var.args.useLoss = false;

    setup.lossFcns.smooth.class = @smoothnessLoss;
    setup.lossFcns.smooth.name = 'Roughness';
    setup.lossFcns.smooth.args.Lambda = 1E-2;
    setup.lossFcns.smooth.args.useLoss = true;

    setup.lossFcns.cls.class = @classifierLoss;
    setup.lossFcns.cls.name = 'Classification';
    setup.lossFcns.cls.args.useLoss = false;

    % model
    setup.model.class = @fcModel;
    setup.model.args.ZDim = 4;
    setup.model.args.isVAE = false;
    setup.model.args.auxModel = 'Fisher';
    
    % training
    setup.trainer.args.updateFreq = 5;
    setup.trainer.args.valType = 'AuxModel';
    setup.trainer.args.nEpochs = 200;
    setup.trainer.args.batchSize = 40;


end