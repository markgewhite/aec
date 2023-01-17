% Run the analysis for the exemplar data sets

clear;

runAnalysis = true;
inParallel = false;
resume = false;
catchErrors = false;
reportIdx = 1:3;
plotDim = [2 5];

% set the destinations for results and figures
path0 = fileparts( which('code/exemplarAnalysis.m') );
path = [path0 '/../results/exemplars_test/'];
pathResults = [path0 '/../paper/results/'];

% -- data setup --
setup.data.class = @ExemplarDataset;   
setup.data.args.HasNormalizedInput = true;

% -- model setup --
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
setup.model.args.trainer.NumIterations = 1;
setup.model.args.trainer.BatchSize = 5000;
setup.model.args.trainer.UpdateFreq = 2000;
setup.model.args.trainer.Holdout = 0;

% --- evaluation setup ---
setup.eval.args.verbose = true;
setup.eval.args.CVType = 'Holdout';

names = [ "Dataset A", ...
          "Dataset B", ...
          "Dataset C" ];
memorySaving = 3;

% -- grid search --
parameters = [ "model.class", "model.args.ZDim" ];
dims = [1 2 3 4];
values = {{@FCModel, @ConvolutionalModel, @PCAModel}, dims}; 
N = 50;
sigma = 0.5;

nReports = length( reportIdx );
nModels = length( values{1} );
nDims = length( dims );

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
                setup.data.args.ClassPeaks = [ 2.0; 1.5 ];
                setup.data.args.ClassPositions = [ 0; 0 ];
                setup.data.args.ClassWidths = [ 1.0; 1.5 ];
        
                setup.data.args.Covariance{1} = 0.1*[1 0 -sigma; 
                                                     0 1 0;
                                                     -sigma 0 1];
                setup.data.args.Covariance{2} = 0.05*[1 0 -sigma; 
                                                     0 1 0;
                                                     -sigma 0 1];

            case 2
                % Data set B: two double-Gaussian classes
                setup.data.args.FeatureType = 'Gaussian';
                setup.data.args.ClassSizes = [ N/2 N/2 ];
                setup.data.args.ClassElements = 2;
                setup.data.args.ClassPeaks = [ 2.0 1.0; 1.0 1.5 ];
                setup.data.args.ClassPositions = [ -1 1; -1 2 ];
                setup.data.args.ClassWidths = [ 1.0 0.2; 0.8 0.1 ];
      
                setup.data.args.Covariance{1} = 0.1*[1 0 -sigma; 
                                                     0 1 0;
                                                     -sigma 0 1];

                setup.data.args.Covariance{2} = 0.05*[1 0 -sigma; 
                                                     0 1 0;
                                                     -sigma 0 1];


            case 3
                % Data set C: two single-sigmoid classes of variable length
                setup.data.args.FeatureType = 'Sigmoid';
                setup.data.args.HasVariableLength = true;
                setup.data.args.TerminationValues = [0.05, 0.95];
                setup.data.args.TerminationTypes = ["Below", "Above"];

                setup.data.args.ClassSizes = [ N/2 N/2 ];
                setup.data.args.ClassElements = 1;
                setup.data.args.ClassPeaks = [ 2.0; 2.0 ];
                setup.data.args.ClassPositions = [ -1; 1 ];
                setup.data.args.ClassWidths = [ 0.5; 0.5 ];
        
                setup.data.args.Covariance{1} = 0.01*[1 0 -sigma; 
                                                     0 1 0;
                                                     -sigma 0 1];
                setup.data.args.Covariance{2} = setup.data.args.Covariance{1};
              
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
                                   memorySaving, resume, catchErrors );
        else
            results(i) = investigationResults( names(i), path, ...
                                               parameters, values, setup, ...
                                               memorySaving, resume, catchErrors );
        end
    
    end

else

    % load from files instead
    for i = reportIdx
        filename = strcat( names(i), "/", names(i), "-Investigation" );
        load( fullfile( path, filename ), 'report' );
        results{i} = report;
    end

    % compile results for the paper
    fields = [ "ReconLossRegular", "AuxModelErrorRate", ...
               "ZCorrelation", "XCCorrelation" ];
    nFields = length( fields );
    
    for d = 1:nDims
    
        for i = 1:nFields
            for j = 1:nModels
                fieldName = strcat( fields(i), num2str(j) );
                T.(fieldName) = zeros( nReports, 1 );
                for k = reportIdx
                    T.(fieldName)(k) = results{k}.TestingResults.Mean.(fields(i))(j,d);
                end
            end
        end
        T0 = struct2table( T );
    
        T0 = genPaperTableCSV( T0, direction = "Rows", criterion = "Smallest", ...
                               groups = [ {1:nModels}, {3*nModels+1:4*nModels} ] );
    
        filename = strcat( "Exemplars-Dim", num2str(d), ".csv" );
        writetable( T0, fullfile( pathResults, filename ) );
    
    end
    
    % save the dataset plots
    genPaperDataPlots( path, "Exemplars", names );
    
    % re-save the component plots
    genPaperCompPlots( path, "Exemplars", names, nDims, nReports, nModels );

end