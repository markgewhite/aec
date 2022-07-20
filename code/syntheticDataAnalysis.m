% Run the analysis for the synthetic data sets

clear;

runAnalysis = true;

% set the destinations for results and figures
path = fileparts( which('code/syntheticDataAnalysis.m') );
path = [path '/../results/test/'];

path2 = fileparts( which('code/syntheticDataAnalysis.m') );
path2 = [path2 '/../paper/results/'];

% -- data setup --
setup.data.class = @SyntheticDataset;
setup.data.args.ClassSizes = [200 200];
setup.data.args.HasNormalizedInput = true;
setup.data.args.OverSmoothing = 1E5;
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
values = { {@FCModel, @ConvolutionalModel, @FullPCAModel}, ...
           seeds }; 

nModels = length( values{1} );
nReports = 4;

names = [ "1L", "2L", "3L", "4L" ];
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
           "AuxModelLoss" ];
%fields = [ "ReconLoss", "ReconLossSmoothed", "ReconLossRegular", ...
%           "ReconBias", "ReconVar", ...
%           "AuxModelLoss", "AuxNetworkLoss", "ComparatorLoss" ];
nFields = length( fields );
groupings = cell( nFields, 1 );
d = 4;
rowNames = [ "2L:S1-C2"; ...
             "3L:S12-C3"; ...
             "3L:S1-C23"; ...
             "4L:S12-C34" ];
for i = 1:nFields

    for j = 1:nModels

        fieldName = strcat( fields(i), num2str(j) );
        T.Mean.(fieldName) = zeros( nReports, 1 );
        T.SD.(fieldName) = zeros( nReports, 1 );

        for k = 1:nReports

            q = zeros( nDatasets, 1 );
            for m = 1:nDatasets
                q(m) = results{k}.TestingResults.(fields(i))(j,d,m);
            end
            T.Mean.(fieldName)(k) = mean(q);
            T.SD.(fieldName)(k) = std(q);

        end
    end

    if i==2
        groupings{i} = (i-1)*nModels+(1:2);
    else
        groupings{i} = (i-1)*nModels+(1:nModels);
    end

end
T0 = struct2table( T.Mean );
T1 = struct2table( T.SD );

T0 = genPaperTableCSV( rowNames, T0, T1, ...
                       direction = "Rows", criterion = "Lowest", ...
                       groups = groupings );

filename = strcat( "Synthetic-Results.csv" );
writetable( T0, fullfile( path2, filename ) );


