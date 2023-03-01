% Run the analysis for how best to create functional components

clear;

runAnalysis = true;
inParallel = false;
resume = false;
catchErrors = false;
reportIdx = 1;
plotDim = [2 5];

% set the destinations for results and figures
path0 = fileparts( which('code/componentAnalysis.m') );
path = [path0 '/../results/test/'];
pathResults = [path0 '/../paper/results/'];

% -- data setup --
setup.data.args.HasNormalizedInput = true;

% -- model setup --
setup.model.class = @FCModel;
setup.model.args.ZDim = 2;
setup.model.args.NumHidden = 1;
setup.model.args.NumFC = 10;
setup.model.args.ReLuScale = 0.2;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.HasCentredDecoder = true;
setup.model.args.RandomSeed = 1234;
setup.model.args.ShowPlots = true;

% -- loss functions --
setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
setup.model.args.lossFcns.recon.name = 'Reconstruction';

setup.model.args.lossFcns.reconrough.class = @ReconstructionRoughnessLoss;
setup.model.args.lossFcns.reconrough.name = 'ReconstructionRoughness';

setup.model.args.lossFcns.zorth.class = @OrthogonalLoss;
setup.model.args.lossFcns.zorth.name = 'ZOrthogonality';

setup.model.args.lossFcns.xorth.class = @ComponentLoss;
setup.model.args.lossFcns.xorth.name = 'XOrthogonality';
setup.model.args.lossFcns.xorth.args.Criterion = 'Orthogonality';
setup.model.args.lossFcns.xorth.args.Alpha = 1E1;

setup.model.args.lossFcns.xvar.class = @ComponentLoss;
setup.model.args.lossFcns.xvar.name = 'XVariance';
setup.model.args.lossFcns.xvar.args.Criterion = 'Varimax';
setup.model.args.lossFcns.xvar.args.Alpha = 1E-1;

% -- trainer setup --
setup.model.args.trainer.NumIterations = 1000;
setup.model.args.trainer.BatchSize = 100;
setup.model.args.trainer.UpdateFreq = 500;
setup.model.args.trainer.Holdout = 0;

% --- evaluation setup ---
setup.eval.args.CVType = 'Holdout';
setup.eval.args.KFolds = 2;
setup.eval.args.KFoldRepeats = 1;

names = [ "JumpsVGRF", ...
          "FukuchiJointAngles" ];
memorySaving = 3;

% -- grid search --
normPts = [11 21];
normTypes = {'None', 'Batch', 'Layer'};
actTypes = {'None', 'Tanh', 'Relu'};
parameters = [ "data.args.NormalizedPts", ...
               "data.args.HasAdaptiveTimeSpan", ...
               "model.args.UsesFdCoefficients", ...
               "model.args.lossFcns.reconrough.args.DoCalcLoss", ...
               "model.args.lossFcns.zorth.args.DoCalcLoss", ...
               "model.args.lossFcns.xorth.args.DoCalcLoss", ...
               "model.args.lossFcns.xvar.args.DoCalcLoss" ];
values = {normPts, ...
          {false, true}, ...
          {false, true}, ...
          {false, true}, ...
          {false, true}, ...
          {false, true}, ...
          {false, true} }; 

nReports = length( reportIdx );

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
                setup.data.args.ResampleRate = 5;
       
            case 2
                % Fukuchi hip, knee and ankle joint angles
                setup.data.class = @FukuchiDataset;
                setup.data.args.YReference = 'AgeGroup';
                setup.data.args.Category = 'JointAngles';
                setup.data.args.HasHipAngles = true;
                setup.data.args.HasKneeAngles = true;
                setup.data.args.HasAnkleAngles = true;
                setup.data.args.SagittalPlaneOnly = true;            
   
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