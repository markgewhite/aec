% Run the analysis of reproducibility using the exemplar data sets

clear;

% set the results destination
path = fileparts( which('code/reproducibilityAnalysis.m') );
path = [path '/../results/reproducibility/'];
attempt = '003';

% -- data setup --
setup.data.class = @ExemplarDataset;   
setup.data.args.HasNormalizedInput = true;
setup.data.args.OverSmoothing = 1E8;

% one class, one element
setup.data.args.ClassSizes = 400;
setup.data.args.ClassElements = 2;
setup.data.args.ClassMeans = [ -1.0 1.0 ];
setup.data.args.ClassSDs = [ 0.5 0.5 ];
setup.data.args.ClassPeaks = [ 2.0 1.0 ];

% double Gaussian with peak inverse covariance
setup.data.args.PeakCovariance{1} = [1 -0.5; -0.5 1];
setup.data.args.MeanCovariance{1} = [1 -0.5; -0.5 1];
setup.data.args.SDCovariance{1} = [1 0.5; 0.5 1];


% -- loss functions --
setup.lossFcns.recon.class = @ReconstructionLoss;
setup.lossFcns.recon.name = 'Reconstruction';

setup.lossFcns.adv.class = @AdversarialLoss;
setup.lossFcns.adv.name = 'Discriminator';

% -- model setup --
setup.model.args.ZDim = 4;
setup.model.args.InitZDimActive = 0;
setup.model.args.KFolds = 10;
setup.model.args.InitZDimActive = 0;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.randomSeed = 1234;

% -- trainer setup --
setup.model.args.trainer.numEpochs = 400;
setup.model.args.trainer.numEpochsPreTrn = 10;
setup.model.args.trainer.updateFreq = 100;
setup.model.args.trainer.batchSize = 50;
setup.model.args.trainer.holdout = 0;

% -- grid search --
parameters = [ "model.class" ];
values = { {@FCModel} };

nTests = 6;
theInvestigation = cell( nTests,1 );

for i = 1:nTests

    switch i

        case 1
            % Full control - identical outputs
            name = [ 'Exemplar-' attempt '-FullControl' ];
            setup.model.args.randomSeedResets = true;
            setup.model.args.identicalNetInit = true;
            setup.model.args.trainer.miniBatchShuffle = false;
            setup.model.args.trainer.randomStatePreservedOnShuffle = false;
            setup.model.args.IdenticalPartitions = true;
       
            theInvestigation{i} = ...
                Investigation( name, path, parameters, values, setup );

        case 2
            % Allow dropout to vary alone
            name = [ 'Exemplar-' attempt '-DropoutVaries' ];
            setup.model.args.randomSeedResets = false;
            setup.model.args.identicalNetInit = true;
            setup.model.args.trainer.miniBatchShuffle = false;
            setup.model.args.trainer.randomStatePreservedOnShuffle = false;
            setup.model.args.IdenticalPartitions = true;

            theInvestigation{i} = ...
                Investigation( name, path, parameters, values, setup );

        case 3
            % Allow mini-batching to vary alone
            name = [ 'Exemplar-' attempt '-MiniBatchVaries' ];
            setup.model.args.randomSeedResets = true;
            setup.model.args.identicalNetInit = true;
            setup.model.args.trainer.miniBatchShuffle = true;
            setup.model.args.trainer.randomStatePreservedOnShuffle = true;
            setup.model.args.IdenticalPartitions = true;
       
            theInvestigation{i} = ...
                Investigation( name, path, parameters, values, setup );

        case 4
            % Allow dropout and mini-batching to vary together
            name = [ 'Exemplar-' attempt '-MiniBatch&DropoutVary' ];
            setup.model.args.randomSeedResets = false;
            setup.model.args.identicalNetInit = true;
            setup.model.args.trainer.miniBatchShuffle = true;
            setup.model.args.trainer.randomStatePreservedOnShuffle = false;
            setup.model.args.IdenticalPartitions = true;
       
            theInvestigation{i} = ...
                Investigation( name, path, parameters, values, setup );
        
        case 5
            % Allow network initialization to vary alone
            name = [ 'Exemplar-' attempt '-NetInitVaries' ];
            setup.model.args.randomSeedResets = true;
            setup.model.args.identicalNetInit = false;
            setup.model.args.trainer.miniBatchShuffle = false;
            setup.model.args.trainer.randomStatePreservedOnShuffle = false;
            setup.model.args.IdenticalPartitions = true;
       
            theInvestigation{i} = ...
                Investigation( name, path, parameters, values, setup );

        case 6
            % Allow data partitions to vary alone
            name = [ 'Exemplar-' attempt '-DataVaries' ];
            setup.model.args.randomSeedResets = true;
            setup.model.args.identicalNetInit = true;
            setup.model.args.trainer.miniBatchShuffle = false;
            setup.model.args.trainer.randomStatePreservedOnShuffle = false;
            setup.model.args.IdenticalPartitions = false;
       
            theInvestigation{i} = ...
                Investigation( name, path, parameters, values, setup );

    end

end