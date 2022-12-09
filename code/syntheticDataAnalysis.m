% Run the analysis for the synthetic data sets

runAnalysis = true;
inParallel = true;
resume = false;

% set the destinations for results and figures
path0 = fileparts( which('code/syntheticDataAnalysis.m') );
path = [path0 '/../results/synthetic3/'];
pathResults = [path0 '/../paper/results/'];

% -- data setup --
setup.data.class = @SyntheticDataset;
setup.data.args.ClassSizes = [200 200];
setup.data.args.HasNormalizedInput = false;
setup.data.args.NumPts = 201;
zscore = 0.5;

% -- model setup --
setup.model.args.ZDim = 4;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.HasCentredDecoder = true;
setup.model.args.RandomSeed = 1234;
setup.model.args.ShowPlots = true;

% -- loss functions --
setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
setup.model.args.lossFcns.recon.name = 'Reconstruction';
setup.model.args.lossFcns.zcls.class = @ClassifierLoss;
setup.model.args.lossFcns.zcls.name = 'ZClassifier';

% -- trainer setup --
setup.model.args.trainer.NumIterations = 5;
setup.model.args.trainer.BatchSize = sum( setup.data.args.ClassSizes );
setup.model.args.trainer.UpdateFreq = 1000;
setup.model.args.trainer.Holdout = 0;

% -- evaluations --
setup.eval.args.verbose = true;
setup.eval.args.CVType = 'Holdout';

parameters = [ "model.class" ];
values = {{@PCAModel, @FCModel, @ConvolutionalModel}};

nModels = length( values{1} );
nReports = 8;

names = [ "1L", "2L", "3L", "4L", "5L", "6L", "7L", "8L" ];
thisData = cell( nReports, 1 );
memorySaving = 4;

if runAnalysis

    if inParallel
        delete( gcp('nocreate') );
        pool = parpool;
    end

    for i = 1:nReports
    
        switch i

            case 1
                % two levels - common & class-specific
                setup.data.args.NumTemplatePts = 5;
                setup.data.args.Scaling = [2 1];
                setup.data.args.Mu = 0.50*[2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0.0;   
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 1;

            case 2
                % three levels - 2 common & 1 class-specific
                setup.data.args.NumTemplatePts = 9;
                setup.data.args.Scaling = [4 2 1];
                setup.data.args.Mu = 0.33*[3 2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0;    
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 2;

            case 3
                % three levels - 1 common & 2 class-specific
                setup.data.args.NumTemplatePts = 9;
                setup.data.args.Scaling = [4 2 1];
                setup.data.args.Mu = 0.33*[3 2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0;    
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 1;

            case 4
                % four levels - 2 common & 2 class-specific
                setup.data.args.NumTemplatePts = 17;
                setup.data.args.Scaling = [8 4 2 1];
                setup.data.args.Mu = 0.25*[4 3 2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0;    
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 2;

            case 5
                % four levels - 3 common & 1 class-specific
                setup.data.args.NumTemplatePts = 17;
                setup.data.args.Scaling = [8 4 2 1];
                setup.data.args.Mu = 0.25*[4 3 2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0;    
                setup.data.args.WarpLevel = 1;
                setup.data.args.SharedLevel = 3;

            case 6
                % four levels - 2 common & 3 class-specific, level 1 warping
                setup.data.args.NumTemplatePts = 17;
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
                setup.data.args.NumTemplatePts = 17;
                setup.data.args.Scaling = [8 4 2 1];
                setup.data.args.Mu = 0.25*[4 3 2 1];
                setup.data.args.Sigma = zscore*setup.data.args.Mu;
                setup.data.args.Eta = 0.1;
                setup.data.args.Tau = 0.4;    
                setup.data.args.WarpLevel = 2;
                setup.data.args.SharedLevel = 3;

        end
    
        if inParallel
            synthResults(i) = parfeval( pool, @investigationResults, 1, ...
                                   names(i), path, ...
                                   parameters, values, setup, ...
                                   memorySaving, resume );
        else
            synthResults(i) = investigationResults( names(i), path, ...
                                               parameters, values, setup, ...
                                               memorySaving, resume );
        end
    
    end

else

    % load from files instead
    for i = 1:nReports
        filename = strcat( names(i), "/", names(i), "-Investigation" );
        load( fullfile( path0, filename ), 'report' );
        synthResults{i} = report;
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
    
T0 = genPaperResultsTable( synthResults, fields, groupSizes );

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
writetable( T0, fullfile( pathResults, filename ) );


