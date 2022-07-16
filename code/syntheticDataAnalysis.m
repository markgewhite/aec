% Run the analysis for the synthetic data sets

clear;

runAnalysis = true;

% set the destinations for results and figures
path = fileparts( which('code/syntheticDataAnalysis.m') );
path = [path '/../results/exemplars/'];

path2 = fileparts( which('code/syntheticDataAnalysis.m') );
path2 = [path2 '/../paper/results/'];

% -- data setup --
setup.data.class = @SyntheticDataset;
setup.data.args.ClassSizes = [250 250];
setup.data.args.HasNormalizedInput = true;
setup.data.args.OverSmoothing = 1E5;

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
setup.model.args.trainer.batchSize = 1000;
setup.model.args.trainer.holdout = 0;

% -- grid search --
nDims = 4;
nDatasets = 10;
parameters = [ "data.args.TemplateSeed", ...
               "model.class", ...
               "model.args.ZDim" ];
seeds = randi( 1000, 1, nDatasets );
values = { seeds, ...
           {@FCModel, @ConvolutionalModel, @FullPCAModel}, ...
           1:nDims }; 
zscore = 0.5;

nModels = length( values{1} );
nReports = 3;

name = [ "1L", ...
         "2L", ...
         "3L" ];
results = cell( nReports, 1 );
thisData = cell( nReports, 1 );

if runAnalysis
    for i = 1:nReports
    
        switch i
    
            case 1
                % one level
                setup.data.args.NumPts = 5;
                setup.data.args.Scaling = 1;
                setup.data.args.Mu = 1;
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0.0;   
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 0;

            case 2
                % two levels
                setup.data.args.NumPts = 5;
                setup.data.args.Scaling = [2 1];
                setup.data.args.Mu = 0.25*[4 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0.0;   
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 1;

            case 3
                % three levels
                setup.data.args.NumPts = 11;
                setup.data.args.Scaling = [4 2 1];
                setup.data.args.Mu = 0.25*[4 2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0;    
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 1;

        end
    
        thisRun = Investigation( name(i), path, parameters, values, setup );
        argsCell = namedargs2cell( setup.data.args );
        thisData{i} = thisRun.getDatasets( which = "First", set = "Testing" );
        results{i} = thisRun.getResults;
    
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
            T.(fieldName) = zeros( nReports, 1 );
            for k = 1:nReports
                T.(fieldName)(k) = results{k}.TestingResults.(fields(i))(j,d);
            end
        end
    end
    T0 = struct2table( T );

    T0 = genPaperTableCSV( T0, direction = "Rows", criterion = "Lowest", ...
                           groups = [ {1:nModels}, {3*nModels+1:4*nModels} ] );

    filename = strcat( "Synthetic-Dim", num2str(d), ".csv" );
    writetable( T0, fullfile( path2, filename ) );

end

% save the dataset plots
genPaperDataPlots( thisData, "Synthetic", name );

% re-save the component plots
genPaperCompPlots( path, "Synthetic", name, 2, nReports, nModels ) 