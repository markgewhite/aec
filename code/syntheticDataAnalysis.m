% Run the analysis for the synthetic data sets

%clear;

runAnalysis = false;

% set the destinations for results and figures
path = fileparts( which('code/syntheticDataAnalysis.m') );
path = [path '/../results/synthetic/'];

path2 = fileparts( which('code/syntheticDataAnalysis.m') );
path2 = [path2 '/../paper/results/'];

% -- data setup --
setup.data.class = @SyntheticDataset;
setup.data.args.ClassSizes = [250 250];
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
setup.model.args.InitZDimActive = 0;
setup.model.args.KFolds = 1;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.randomSeed = 1234;

% -- trainer setup --
setup.model.args.trainer.numEpochs = 400;
setup.model.args.trainer.numEpochsPreTrn = 10;
setup.model.args.trainer.updateFreq = 100;
setup.model.args.trainer.batchSize = 100;
setup.model.args.trainer.holdout = 0;

% -- grid search --
nDims = 4;
nDatasets = 5;
rng( setup.model.args.randomSeed );
seeds = randi( 1000, 1, nDatasets );

parameters = [ "model.class", ...
               "model.args.ZDim", ...
               "data.args.TemplateSeed" ];
values = { {@FCModel, @ConvolutionalModel, @FullPCAModel}, ...
           1:nDims, ...
           seeds }; 

nModels = length( values{1} );
nReports = 4;

name = [ "1L", "2L", "3L", "4L" ];
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
    
        thisRun = Investigation( name(i), path, ...
                                 parameters, values, setup, memorySaving );
        results{i} = thisRun.getResults;
        clear thisRun;
    
    end

else

    % load from files instead
    for i = 1:nReports
        filename = strcat( name(i), "/", name(i), "-Investigation" );
        load( fullfile( path, filename ), 'report' );
        results{i} = report;
    end

end

% compile results for the paper
fields = [ "ReconLossRegular", "AuxModelLoss", ...
           "ZCorrelation", "XCCorrelation" ];
nFields = length( fields );

for d = 1:nDims

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
    end
    T0 = struct2table( T.Mean );
    T1 = struct2table( T.SD );

    T0 = genPaperTableCSV( T0, T1, ...
                           direction = "Rows", criterion = "Lowest", ...
                           groups = [ {1:nModels}, ...
                                      {nModels+1:2*nModels}, ...
                                      {2*nModels+1:3*nModels}, ...
                                      {3*nModels+1:4*nModels} ] );

    filename = strcat( "Synthetic-Dim", num2str(d), ".csv" );
    writetable( T0, fullfile( path2, filename ) );

end

