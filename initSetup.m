function setup = initSetup
    % Specify the configuration where setting differ from default values

    % dataset
    setup.data.class = @jumpGRFDataset;
    setup.data.args.normalization = 'PAD';
    setup.data.args.normalizeInput = false;

    % loss functions
    setup.lossFcns.recon.class = @reconstructionLoss;
    setup.lossFcns.recon.name = 'Reconstruction';

    %setup.lossFcns.adv.class = @adversarialLoss;
    %setup.lossFcns.adv.name = 'Discriminator';

    %setup.lossFcns.cls.class = @classifierLoss;
    %setup.lossFcns.cls.name = 'JumpType';

    %setup.lossFcns.mmd.class = @wassersteinLoss;
    %setup.lossFcns.mmd.name = 'MMDDiscriminator';
    %setup.lossFcns.mmd.args.kernel = 'IMQ';
    %setup.lossFcns.mmd.args.useLoss = false;

    %setup.lossFcns.orth.class = @componentLoss;
    %setup.lossFcns.orth.name = 'Component';
    %setup.lossFcns.orth.args.nSample = 10;
    %setup.lossFcns.orth.args.criterion = 'Orthogonality';

    setup.lossFcns.smooth.class = @smoothnessLoss;
    setup.lossFcns.smooth.name = 'Roughness';
    setup.lossFcns.smooth.args.Lambda = 1E-2;
    setup.lossFcns.smooth.args.useLoss = true;

    % model
    setup.model.class = @lstmfcModel;
    setup.model.args.ZDim = 4;
    setup.model.args.isVAE = false;
    setup.model.args.auxModel = 'Fisher';
    setup.model.args.inputDropout = 0.0;
    setup.model.args.nLSTMHidden = 4;
    setup.model.args.lstmFactor = -1;
    setup.model.args.bidirectional = false;
    setup.model.args.scale = 0.2;

    % training
    setup.trainer.args.updateFreq = 5;
    setup.trainer.args.valType = 'AuxModel';
    setup.trainer.args.nEpochs = 100;


end