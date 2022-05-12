function setup = initSetup
    % Specify the configuration where setting differ from default values

    % dataset
    setup.data.class = @exemplarDataset;
    setup.data.args.Type = 'Gaussian';
    setup.data.args.ClassSizes = 500;
    setup.data.args.normalization = 'PAD';
    setup.data.args.normalizeInput = false;
    setup.data.args.normalizedPts = 51;
    setup.data.args.adaptiveTimeSpan = true;
    %setup.data.args.resampleRate = 10;

    % loss functions
    setup.lossFcns.recon.class = @reconstructionLoss;
    setup.lossFcns.recon.name = 'Reconstruction';

    setup.lossFcns.adv.class = @adversarialLoss;
    setup.lossFcns.adv.name = 'Discriminator';

    setup.lossFcns.cls.class = @classifierLoss;
    setup.lossFcns.cls.name = 'Classification';

    %setup.lossFcns.mmd.class = @wassersteinLoss;
    %setup.lossFcns.mmd.name = 'MMDDiscriminator';
    %setup.lossFcns.mmd.args.kernel = 'IMQ';
    %setup.lossFcns.mmd.args.useLoss = false;

    setup.lossFcns.orth.class = @componentLoss;
    setup.lossFcns.orth.name = 'Component';
    setup.lossFcns.orth.args.nSample = 10;
    setup.lossFcns.orth.args.criterion = 'Orthogonality';

    setup.lossFcns.smooth.class = @smoothnessLoss;
    setup.lossFcns.smooth.name = 'Roughness';
    setup.lossFcns.smooth.args.Lambda = 1E-2;
    setup.lossFcns.smooth.args.useLoss = false;

    % model
    setup.model.class = @fcModel;
    setup.model.args.ZDim = 4;
    setup.model.args.isVAE = false;
    setup.model.args.auxModel = 'Fisher';
    
    % training
    setup.trainer.args.updateFreq = 20;
    setup.trainer.args.valType = 'AuxModel';
    setup.trainer.args.nEpochs = 200;
    setup.trainer.args.batchSize = 100;


end