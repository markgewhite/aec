% Run the analysis for the exemplar data sets

clear;

runAnalysis = true;
inParallel = true;
resume = false;
catchErrors = true;
reportIdx = 1:4;
plotDim = [2 5];

% set the destinations for results and figures
path0 = fileparts( which('code/exemplarAnalysis.m') );
path = [path0 '/../results/exemplars/'];
pathResults = [path0 '/../paper/results/'];

% -- data setup --


% -- model setup --
setup.model.class = @FCModel;
setup.model.args.NumHidden = 1;
setup.model.args.NumFC = 100;
setup.model.args.FCFactor = 1;       
setup.model.args.ReLuScale = 1;
setup.model.args.InputDropout = 0;
setup.model.args.Dropout = 0;
setup.model.args.HasBatchNormalization = false;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.HasCentredDecoder = true;
setup.model.args.RandomSeed = 1234;
setup.model.args.ShowPlots = true;

% -- loss functions --
setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
setup.model.args.lossFcns.recon.name = 'Reconstruction';
setup.model.args.lossFcns.zcls.class = @ClassifierLoss;
setup.model.args.lossFcns.zcls.name = 'ZClassifier';
setup.model.args.lossFcns.zcls.args.NumHidden = 1;
setup.model.args.lossFcns.zcls.args.NumFC = 100;
setup.model.args.lossFcns.zcls.args.FCFactor = 1;
setup.model.args.lossFcns.zcls.args.ReLuScale = 1;
setup.model.args.lossFcns.zcls.args.Dropout = 0;
setup.model.args.lossFcns.zcls.args.HasBatchNormalization = false;

% -- trainer setup --
setup.model.args.trainer.NumIterations = 2000;
setup.model.args.trainer.BatchSize = 100;
setup.model.args.trainer.UpdateFreq = 5000;
setup.model.args.trainer.Holdout = 0;

% --- evaluation setup ---
setup.eval.args.CVType = 'Holdout';

names = [ "JumpsVGRF", ...
          "Dataset A", ...
          "Dataset B", ...
          "Dataset C" ];
memorySaving = 3;

% -- grid search --
dims = [2 3];
pts = [5, 7, 9, 15 20];
parameters = [ "model.args.ZDim", ...
               "data.args.NormalizedPts", ...
               "model.args.lossFcns.zcls.args.DoCalcLoss"];
values = {dims, pts, {false, true}}; 

N = 400;
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
                % JumpsVGRF data set
                setup.data.class = @JumpGRFDataset;
                setup.data.args.Normalization = 'PAD';
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.ResampleRate = 5;
                setup.data.args.NormalizedPts = 25;
       
            case 2
                % Data set A: two single-Gaussian classes
                setup.data.class = @ExemplarDataset;   
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.NormalizedPts = 11;

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

            case 3
                % Data set B: two double-Gaussian classes
                setup.data.class = @ExemplarDataset;   
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.NormalizedPts = 11;

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


            case 4
                % Data set C: two single-sigmoid classes of variable length
                setup.data.class = @ExemplarDataset;   
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.NormalizedPts = 11;

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
              
   
        end
    
        if inParallel
            results(i) = parfeval( pool, @investigationResults, 1, ...
                                   names(i), path, ...
                                   parameters, values, setup, ...
                                   resume, catchErrors, memorySaving );
        else
            results(i) = investigationResults( names(i), path, ...
                                               parameters, values, setup, ...
                                               resume, catchErrors, memorySaving );
        end
    
    end

else

    % load from files instead
    for i = reportIdx
        filename = strcat( names(i), "-InvestigationReport" );
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

% Code the execute when the parallel processing is complete 
%diary = strings(length(reportIdx),1); 
%for i = reportIdx 
%    diary(i) = results(i).Diary; 
%end
%save( fullfile(path, 'diaries'), 'diary' );