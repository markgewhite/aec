function setup = initSetup
    % Specify the configuration where setting differ from default values

    % dataset
    setup.data.class = @jumpGRFDataset;

    % loss functions
    setup.lossFcns.recon.class = @reconstructionLoss;
    setup.lossFcns.recon.name = 'Reconstruction';

    %setup.lossFcns.adv.class = @adversarialLoss;
    %setup.lossFcns.adv.name = 'Discriminator';

    %setup.lossFcns.cls.class = @classifierLoss;
    %setup.lossFcns.cls.name = 'JumpType';

    setup.lossFcns.mmd.class = @wassersteinLoss;
    setup.lossFcns.mmd.name = 'MMDDiscriminator';
    setup.lossFcns.mmd.args.kernel = 'IMQ';
    setup.lossFcns.mmd.args.useLoss = true;

    %setup.lossFcns.orth.class = @componentLoss;
    %setup.lossFcns.orth.name = 'Component';
    %setup.lossFcns.orth.args.nSample = 10;
    %setup.lossFcns.orth.args.criterion = 'Orthogonality';

    % model
    setup.model.class = @fcModel;
    setup.model.args.ZDim = 4;
    setup.model.args.isVAE = true;
    setup.model.args.auxModel = 'Fisher';

    % training
    setup.trainer.args.updateFreq = 25;
    setup.trainer.args.showPlots = true;
    setup.trainer.args.valType = 'AuxModel';
    setup.trainer.args.nEpochs = 25;


end