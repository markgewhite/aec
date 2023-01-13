% Run the analysis for the exemplar data sets

clear;

runAnalysis = true;
inParallel = false;
resume = false;
reportIdx = 3;
plotDim = [2 5];

% set the destinations for results and figures
path0 = fileparts( which('code/exemplarAnalysis.m') );
path = [path0 '/../results/exemplars/'];
pathResults = [path0 '/../paper/results/'];

% -- data setup --
setup.data.class = @ExemplarDataset;   
setup.data.args.HasNormalizedInput = true;

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
setup.model.args.trainer.NumIterations = 5; %500
setup.model.args.trainer.BatchSize = 5000;
setup.model.args.trainer.UpdateFreq = 2000;
setup.model.args.trainer.Holdout = 0;

% --- evaluation setup ---
setup.eval.args.verbose = true;
setup.eval.args.CVType = 'Holdout';

names = [ "Dataset A", ...
          "Dataset B", ...
          "Dataset C" ];
nReports = length( names );
thisData = cell( nReports, 1 );
memorySaving = 4;

% -- grid search --
nDims = 4;
parameters = [ "model.class" ];
values = {{@FCModel}}; % {{@FCModel, @ConvolutionalModel, @PCAModel}}; 
N = 500;
sigma = 0.8;

nDatasets = 6;
nModels = length( values{1} );
nReports = 8;

name = [ "1G-PeakVar", "1G-MeanVar", ...
         "2G-PeakVar", "2G-MeanSDVar", ...
         "1G-2Classes", "1G-2Classes-Classify", ...
         "2G-2Classes", "2G-2Classes-Classify" ];
results = cell( nReports, 1 );
thisData = cell( nReports, 1 );

if runAnalysis

    if inParallel
        delete( gcp('nocreate') );
        pool = parpool;
    end

    for i = reportIdx
    
        switch i
       
            case 1
                % Data set A: two single-Gaussian classes
                setup.data.args.FeatureType = 'Gaussian';
                setup.data.args.ClassSizes = [ N/2 N/2 ];
                setup.data.args.ClassElements = 1;
                setup.data.args.ClassMeans = [ -1; 1 ];
                setup.data.args.ClassSDs = [ 0.25; 0.5 ];
                setup.data.args.ClassPeaks = [ 2.0; 1.0 ];
        
                setup.data.args.PeakCovariance{1} = sigma;
                setup.data.args.MeanCovariance{1} = sigma;
                setup.data.args.SDCovariance{1} = sigma;
    
                setup.data.args.PeakCovariance{2} = sigma;
                setup.data.args.MeanCovariance{2} = sigma;
                setup.data.args.SDCovariance{2} = sigma;
    
            case 2
                % Data set B: two double-Gaussian classes
                setup.data.args.FeatureType = 'Gaussian';
                setup.data.args.ClassSizes = [ N/2 N/2 ];
                setup.data.args.ClassElements = 2;
                setup.data.args.ClassMeans = [ -0.5 0; 0 0.5 ];
                setup.data.args.ClassSDs = [ 0.5 0.3; 0.2 0.1 ];
                setup.data.args.ClassPeaks = [ 2.0 1.5; 2.0 1.0 ];
        
                setup.data.args.PeakCovariance{1} = 0.1*[1 -sigma; -sigma 1];
                setup.data.args.MeanCovariance{1} = 0.1*[1 -sigma; -sigma 1];
                setup.data.args.SDCovariance{1} = 0.1*[1 -sigma; -sigma 1];
    
                setup.data.args.PeakCovariance{2} = 0.2*[1 -sigma; -sigma 1];
                setup.data.args.MeanCovariance{2} = 0.2*[1 -sigma; -sigma 1];
                setup.data.args.SDCovariance{2} = 0.2*[1 -sigma; -sigma 1];

            case 3
                % Data set C: two single-sigmoid classes of variable length
                setup.data.args.FeatureType = 'Sigmoid';
                setup.data.args.HasVariableLength = true;
                setup.data.args.TerminationValues = [0.05, 0.95];
                setup.data.args.TerminationTypes = ["Below", "Above"];

                setup.data.args.ClassSizes = [ N/2 N/2 ];
                setup.data.args.ClassElements = 1;
                setup.data.args.ClassMeans = [ 0; 0 ];
                setup.data.args.ClassSDs = [ 0.25; 0.5 ];
                setup.data.args.ClassPeaks = [ 1.0; 1.0 ];
        
                setup.data.args.PeakCovariance{1} = sigma;
                setup.data.args.MeanCovariance{1} = sigma;
                setup.data.args.SDCovariance{1} = sigma;
    
                setup.data.args.PeakCovariance{2} = sigma;
                setup.data.args.MeanCovariance{2} = sigma;
                setup.data.args.SDCovariance{2} = sigma;
              
            case 4
                % Double Gaussian with peak inverse covariance
                setup.data.args.PeakCovariance{1} = [1 -sigma; -sigma 1];
                setup.data.args.MeanCovariance{1} = 1E-6*eye(2);
                setup.data.args.SDCovariance{1} = 1E-6*eye(2);
           
                setup.lossFcns = lossFcns1; 
    
        end
    
        if inParallel
            results(i) = parfeval( pool, @investigationResults, 1, ...
                                   names(i), path, ...
                                   parameters, values, setup, ...
                                   memorySaving, resume );
        else
            results(i) = investigationResults( names(i), path, ...
                                               parameters, values, setup, ...
                                               memorySaving, resume );
        end
    
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

    filename = strcat( "Exemplars-Dim", num2str(d), ".csv" );
    writetable( T0, fullfile( path2, filename ) );

end

% save the dataset plots
genPaperDataPlots( thisData, "Exemplars", name );

% re-save the component plots
genPaperCompPlots( path, "Exemplars", name, 2, nReports, nModels ) 