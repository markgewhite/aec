% Run the analysis for the exemplar data sets

clear;

runAnalysis = true;
inParallel = false;
catchErrors = false;
reportIdx = 1;
plotDim = [2 5];

% set the destinations for results and figures
path0 = fileparts( which('code/exemplarAnalysis.m') );
path = [path0 '/../results/test/'];
pathResults = [path0 '/../paper/results/'];

% -- model setup --
setup.model.class = @ConvolutionalModel;
setup.model.args.ZDim = 2;
setup.model.args.NumHidden = 2;
%setup.model.args.FilterSize = 5;
%setup.model.args.Padding = 'None';
setup.model.args.NumHiddenDecoder = 2;
%setup.model.args.FilterSizeDecoder = 17;
%setup.model.args.PaddingDecoder = 'None';
setup.model.args.InputDropout = 0;
setup.model.args.Dropout = 0;
setup.model.args.NetNormalizationType = 'None';
setup.model.args.NetActivationType = 'None';

%setup.model.args.NumHiddenDecoder = 2;
%setup.model.args.NumFCDecoder = 10;
%setup.model.args.FCFactorDecoder = 0;
%setup.model.args.NetNormalizationTypeDecoder = 'None';
%setup.model.args.NetActivationTypeDecoder = 'None';

setup.model.args.ComponentType = 'PDP';
setup.model.args.AuxModel = 'Logistic';
setup.model.args.randomSeed = 1234;
setup.model.args.HasCentredDecoder = true;
setup.model.args.ShowPlots = true;


% -- loss functions --
setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
setup.model.args.lossFcns.recon.name = 'Reconstruction';

setup.model.args.lossFcns.reconrough.class = @ReconstructionRoughnessLoss;
setup.model.args.lossFcns.reconrough.name = 'ReconstructionRoughness';
setup.model.args.lossFcns.reconrough.args.Dilations = 1;

setup.model.args.lossFcns.zorth.class = @OrthogonalLoss;
setup.model.args.lossFcns.zorth.name = 'ZOrthogonality';

setup.model.args.lossFcns.xvar.class = @ComponentLoss;
setup.model.args.lossFcns.xvar.name = 'XVarimax';
setup.model.args.lossFcns.xvar.args.Criterion = 'Varimax';

setup.model.args.lossFcns.zcls.class = @ClassifierLoss;
setup.model.args.lossFcns.zcls.name = 'ZClassifier';
setup.model.args.lossFcns.zcls.args.NumHidden = 1;
setup.model.args.lossFcns.zcls.args.NumFC= 10;
setup.model.args.lossFcns.zcls.args.HasBatchNormalization = false;
setup.model.args.lossFcns.zcls.args.ReluScale = 0;
setup.model.args.lossFcns.zcls.args.Dropout = 0;

% -- trainer setup --
setup.model.args.trainer.NumIterations = 1000;
setup.model.args.trainer.BatchSize = 100;
setup.model.args.trainer.UpdateFreq = 500;
setup.model.args.trainer.Holdout = 0;

% --- evaluation setup ---
setup.eval.args.CVType = 'Holdout';
setup.eval.args.KFolds = 2;
setup.eval.args.KFoldRepeats = 5;
setup.eval.args.InParallel = inParallel;

memorySaving = 3;

% -- grid search --
parameters = [ "model.class", ...
               "model.args.lossFcns.zcls.args.DoCalcLoss"];
values = {{@ConvBranchedModel}, ...
          {false, true}}; 

%parameters = [ "model.args.NetNormalizationType", ...
%               "model.args.NetActivationType", ...
%               "model.args.NetNormalizationTypeDecoder", ...
%               "model.args.NetActivationTypeDecoder" ];
%values = {{'None', 'Batch'}, ...
%          {'None', 'Tanh', 'Relu'}, ...
%          {'None', 'Batch', 'Layer'}, ...
%          {'None', 'Tanh', 'Relu'}}; 

%parameters = [ "model.args.NetNormalizationTypeDecoder", ...
%               "model.args.NetActivationTypeDecoder" ];
%values = {{'None', 'Batch', 'Layer'}, ...
%          {'None', 'Tanh', 'Relu'}}; 


N = 1000;
sigma = 0.5;

myInvestigations = cell( length(reportIdx), 1 );

if runAnalysis

    for i = reportIdx
    
        if isfield(setup, 'data')
            % reset the data settings
            setup = rmfield( setup, 'data' );
        end

        switch i

            case 1
                % Data set A: single Gaussian classes
                name = 'Dataset A';
                setup.data.class = @ExemplarDataset;   
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.normalizedPts = 17;

                setup.data.args.FeatureType = 'Gaussian';
                setup.data.args.ClassSizes = [ N/2 N/2 ];
                setup.data.args.ClassElements = 1;
                setup.data.args.ClassPeaks = [ 2.50; 2.75 ];
                setup.data.args.ClassPositions = [ 0; 0];
                setup.data.args.ClassWidths = [ 2.5; 2.5 ];
                setup.data.args.Covariance{1} = 1E-1*[1 0 0; 
                                                      0 0 0;
                                                      0 0 0];                                                 
                setup.data.args.Covariance{2} = 1E-1*[1 0 0; 
                                                      0 0 0;
                                                      0 0 0];                                                 


            case 2
                % Data set B: single Gaussian classes
                name = 'Dataset B';
                setup.data.class = @ExemplarDataset;   
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.normalizedPts = 17;

                setup.data.args.FeatureType = 'Gaussian';
                setup.data.args.ClassSizes = [ N/2 N/2 ];
                setup.data.args.ClassElements = 1;
                setup.data.args.ClassPeaks = [ 2.50; 2.50 ];
                setup.data.args.ClassPositions = [ -0.25; 0.25 ];
                setup.data.args.ClassWidths = [ 2.5; 2.5 ];
                setup.data.args.Covariance{1} = 1E-1*[0 0 0; 
                                                      0 1 0;
                                                      0 0 0];                                                 
                setup.data.args.Covariance{2} = 1E-1*[0 0 0; 
                                                      0 1 0;
                                                      0 0 0];    

            case 3
                % Data set C: single Gaussian classes
                name = 'Dataset C';
                setup.data.class = @ExemplarDataset;   
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.normalizedPts = 17;

                setup.data.args.FeatureType = 'Gaussian';
                setup.data.args.ClassSizes = [ N/2 N/2 ];
                setup.data.args.ClassElements = 1;
                setup.data.args.ClassPeaks = [ 2.50; 2.50 ];
                setup.data.args.ClassPositions = [ 0; 0 ];
                setup.data.args.ClassWidths = [ 3.0; 2.5 ];
                setup.data.args.Covariance{1} = 1E-1*[0 0 0; 
                                                      0 0 0;
                                                      0 0 1];                                                 
                setup.data.args.Covariance{2} = 1E-1*[0 0 0; 
                                                      0 0 0;
                                                      0 0 1];  

            case 4
                % Data set A: two double-Gaussian classes
                setup.data.class = @ExemplarDataset;   
                setup.data.args.HasNormalizedInput = true;

                setup.data.args.FeatureType = 'Gaussian';
                setup.data.args.ClassSizes = [ N/2 N/2 ];
                setup.data.args.ClassElements = 2;
                setup.data.args.ClassPeaks = [ 2.50 0.0; 2.75 1.0 ];
                setup.data.args.ClassPositions = [ -1 1.5; -1 2 ];
                setup.data.args.ClassWidths = [ 3.0 0.50; 2.5 1.0 ];
      
                setup.data.args.Covariance{1} = 0.1*[1 0 -sigma; 
                                                     0 1 0;
                                                     -sigma 0 1];

                setup.data.args.Covariance{2} = 0.05*[1 0 -sigma; 
                                                     0 1 0;
                                                     -sigma 0 1];             
   
        end

        myInvestigations{i} = Investigation( name, path, parameters, values, ...
                                         setup, catchErrors, memorySaving );
        
        myInvestigations{i}.run;
        
        myInvestigations{i}.saveReport;
        
        myInvestigations{i}.save;

   
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