% Run the analysis for the synthetic data sets

clear;

runAnalysis = true;

% set the destinations for results and figures
path = fileparts( which('code/syntheticDataAnalysis.m') );
path = [path '/../results/synthetic3/'];

path2 = fileparts( which('code/syntheticDataAnalysis.m') );
path2 = [path2 '/../paper/results/'];

% -- data setup --
setup.data.class = @SyntheticDataset;
setup.data.args.ClassSizes = [200 200];
setup.data.args.HasNormalizedInput = false;
setup.data.args.NumPts = 201;
zscore = 0.5;

% -- loss functions --
setup.lossFcns.recon.class = @ReconstructionLoss;
setup.lossFcns.recon.name = 'Reconstruction';
setup.lossFcns.adv.class = @AdversarialLoss;
setup.lossFcns.adv.name = 'Discriminator';
setup.lossFcns.zcls.class = @ClassifierLoss;
setup.lossFcns.zcls.name = 'ZClassifier';
setup.lossFcns.xcls.class = @InputClassifierLoss;
setup.lossFcns.xcls.name = 'XClassifier';

% -- model setup --
setup.model.args.ZDim = 4;
setup.model.args.InitZDimActive = 0;
setup.model.args.KFolds = 1;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.randomSeed = 1234;
setup.model.args.CompressionLevel = 3;

% -- trainer setup --
setup.model.args.trainer.numEpochs = 400; % 400
setup.model.args.trainer.numEpochsPreTrn = 10; %10
setup.model.args.trainer.updateFreq = 200;
setup.model.args.trainer.batchSize = 64;
setup.model.args.trainer.holdout = 0.2;

% -- grid search --
nDatasets = 10;
rng( setup.model.args.randomSeed );
seeds = randi( 1000, 1, nDatasets );

parameters = [ "model.class", ...
               "data.args.TemplateSeed" ];
%values = { {@FCModel, @ConvolutionalModel, @FullPCAModel}, ...
%           seeds }; 
values = { {@TCNModel}, ...
           seeds }; 

nModels = length( values{1} );
nReports = 8;

names = [ "1L", "2L", "3L", "4L", "5L", "6L", "7L", "8L" ];
results = cell( nReports, 1 );
thisData = cell( nReports, 1 );
memorySaving = 4;

if runAnalysis
    for i = 1:nReports
    
        switch i

            case 1
                % two levels - common & class-specific
                setup.data.args.NumPts = 5;
                setup.data.args.Scaling = [2 1];
                setup.data.args.Mu = 0.50*[2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0.0;   
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 1;

            case 2
                % three levels - 2 common & 1 class-specific
                setup.data.args.NumPts = 9;
                setup.data.args.Scaling = [4 2 1];
                setup.data.args.Mu = 0.33*[3 2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0;    
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 2;

            case 3
                % three levels - 1 common & 2 class-specific
                setup.data.args.NumPts = 9;
                setup.data.args.Scaling = [4 2 1];
                setup.data.args.Mu = 0.33*[3 2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0;    
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 1;

            case 4
                % four levels - 2 common & 2 class-specific
                setup.data.args.NumPts = 17;
                setup.data.args.Scaling = [8 4 2 1];
                setup.data.args.Mu = 0.25*[4 3 2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0;    
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 2;

            case 5
                % four levels - 3 common & 1 class-specific
                setup.data.args.NumPts = 17;
                setup.data.args.Scaling = [8 4 2 1];
                setup.data.args.Mu = 0.25*[4 3 2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0;    
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 3;

            case 6
                % four levels - 2 common & 3 class-specific, level 1 warping
                setup.data.args.NumPts = 17;
                setup.data.args.Scaling = [8 4 2 1];
                setup.data.args.Mu = 0.25*[4 3 2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0.2;    
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 3;

            case 7
                % four levels - 2 common & 3 class-specific, level 2 warping
                setup.data.args.NumPts = 17;
                setup.data.args.Scaling = [8 4 2 1];
                setup.data.args.Mu = 0.25*[4 3 2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0.2;    
                setup.data.args.WarpLevel = 2;
                setup.data.args.SharedLevel = 3;

            case 8
                % four levels - 2 common & 3 class-specific, level 3 warping
                setup.data.args.NumPts = 17;
                setup.data.args.Scaling = [8 4 2 1];
                setup.data.args.Mu = 0.25*[4 3 2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0.4;    
                setup.data.args.WarpLevel = 2;
                setup.data.args.SharedLevel = 3;

        end
    
        thisRun = Investigation( names(i), path, ...
                                 parameters, values, setup, memorySaving );
        results{i} = thisRun.getResults;
        clear thisRun;
    
    end

else

    % load from files instead
    for i = 1:nReports
        filename = strcat( names(i), "/", names(i), "-Investigation" );
        load( fullfile( path, filename ), 'report' );
        results{i} = report;
    end

end

% compile results for the paper
fields = [ "ReconLoss", "ReconLossSmoothed", "ReconLossRegular", ...
           "ReconBias", "ReconVar", ...
           "AuxModelLoss", "AuxNetworkLoss", "ComparatorLoss", ...
           "ZCorrelation", "XCCorrelation" ];

groupSizes = [ 3, 3, 3, ...
               2, 2, ...
               3, 2, 2, ...
               3, 3 ];
    
T0 = genPaperResultsTable( results, fields, groupSizes );

TestNames = [ "2L:S1-C2"; ...
             "3L:S12-C3"; ...
             "3L:S1-C23"; ...
             "4L:S12-C34"; ...
             "4L:S123-C4"; ...
             "4L:S123-C4-W1"; ...
             "4L:S123-C4-W2"; ...
             "4L:S123-C4-W2*" ];

T0 = addvars( T0, TestNames, Before = 1 );

filename = strcat( "Synthetic3-Results.csv" );
writetable( T0, fullfile( path2, filename ) );


