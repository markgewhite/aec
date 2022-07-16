% Run the analysis for the exemplar data sets

clear;

runAnalysis = false;

% set the destinations for results and figures
path = fileparts( which('code/exemplarAnalysis.m') );
path = [path '/../results/exemplars/'];

path2 = fileparts( which('code/exemplarAnalysis.m') );
path2 = [path2 '/../paper/results/'];

% -- data setup --
setup.data.class = @ExemplarDataset;   
setup.data.args.HasNormalizedInput = true;
setup.data.args.OverSmoothing = 1E8;

% -- loss functions --
lossFcns1.recon.class = @ReconstructionLoss;
lossFcns1.recon.name = 'Reconstruction';
lossFcns1.adv.class = @AdversarialLoss;
lossFcns1.adv.name = 'Discriminator';

lossFcns2 = lossFcns1;
lossFcns2.zcls.class = @ClassifierLoss;
lossFcns2.zcls.name = 'ZClassifier';
lossFcns2.xcls.class = @InputClassifierLoss;
lossFcns2.xcls.name = 'XClassifier';

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
parameters = [ "model.class" "model.args.ZDim" ];
values = {{@FCModel, @ConvolutionalModel, @FullPCAModel} 1:nDims }; 
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
    for i = 1:nReports
    
        switch i
    
            case {1 2}
                % one class, one element
                setup.data.args.ClassSizes = N;
                setup.data.args.ClassElements = 1;
                setup.data.args.ClassMeans = 0.0;
                setup.data.args.ClassSDs = 0.5;
                setup.data.args.ClassPeaks = 2.0;
    
            case {3 4}
                % one class, two elements
                setup.data.args.ClassSizes = N;
                setup.data.args.ClassElements = 2;
                setup.data.args.ClassMeans = [ -1.0 1.0 ];
                setup.data.args.ClassSDs = [ 0.5 0.5 ];
                setup.data.args.ClassPeaks = [ 2.0 1.0 ];
    
            case {5 6}
                % Two classes each with a single Gaussian
                setup.data.args.ClassSizes = [ N/2 N/2 ];
                setup.data.args.ClassElements = 1;
                setup.data.args.ClassMeans = [ -1; 1 ];
                setup.data.args.ClassSDs = [ 0.5; 0.5 ];
                setup.data.args.ClassPeaks = [ 2.0; 1.0 ];
        
                setup.data.args.PeakCovariance{1} = 1;
                setup.data.args.MeanCovariance{1} = 1E-6;
                setup.data.args.SDCovariance{1} = 1;
    
                setup.data.args.PeakCovariance{2} = sigma;
                setup.data.args.MeanCovariance{2} = 1E-6;
                setup.data.args.SDCovariance{2} = 1E-6;
    
            case {7 8}
                % Two classes each with a double Gaussian
                setup.data.args.ClassSizes = [ N/2 N/2 ];
                setup.data.args.ClassElements = 2;
                setup.data.args.ClassMeans = [ -1 0; 0 1 ];
                setup.data.args.ClassSDs = [ 0.5 0.3; 0.2 0.1 ];
                setup.data.args.ClassPeaks = [ 2.0 3.0; 2.0 1.0 ];
        
                setup.data.args.PeakCovariance{1} = 0.1*[1 -sigma; -sigma 1];
                setup.data.args.MeanCovariance{1} = 0.1*[1 -sigma; -sigma 1];
                setup.data.args.SDCovariance{1} = 0.1*[1 -sigma; -sigma 1];
    
                setup.data.args.PeakCovariance{2} = 0.2*[1 -sigma; -sigma 1];
                setup.data.args.MeanCovariance{2} = 0.2*[1 -sigma; -sigma 1];
                setup.data.args.SDCovariance{2} = 0.2*[1 -sigma; -sigma 1];
        
        end
    
        switch i
    
            case 1
                % Single Gaussian with peak (height) variance   
                setup.data.args.PeakCovariance{1} = 1;
                setup.data.args.MeanCovariance{1} = 1E-6;
                setup.data.args.SDCovariance{1} = 1E-6;
           
                setup.lossFcns = lossFcns1;
    
            case 2
                % Single Gaussian with mean (position) variance
                setup.data.args.PeakCovariance{1} = 1E-6;
                setup.data.args.MeanCovariance{1} = 1;
                setup.data.args.SDCovariance{1} = 1E-6;
        
                setup.lossFcns = lossFcns1;
    
            case 3
                % Double Gaussian with peak inverse covariance
                setup.data.args.PeakCovariance{1} = [1 -sigma; -sigma 1];
                setup.data.args.MeanCovariance{1} = 1E-6*eye(2);
                setup.data.args.SDCovariance{1} = 1E-6*eye(2);
           
                setup.lossFcns = lossFcns1;
    
            case 4
                % Double Gaussian with peak inverse covariance   
                setup.data.args.PeakCovariance{1} = [1 sigma; sigma 1];
                setup.data.args.MeanCovariance{1} = [1 -sigma; -sigma 1];
                setup.data.args.SDCovariance{1} = [1 sigma; sigma 1];
           
                setup.lossFcns = lossFcns1;
    
            case 5
                % Two classes each with a single Gaussian
                setup.lossFcns = lossFcns1;
    
            case 6
                % Two classes each with a single Gaussian - with classification
                setup.lossFcns = lossFcns2;       
    
            case 7
                % Two classes each with a double Gaussian
                setup.lossFcns = lossFcns1;
    
            case 8
                % Two classes each with a double Gaussian - with classification
                setup.lossFcns = lossFcns2;       
    
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

    filename = strcat( "Exemplars-Dim", num2str(d), ".csv" );
    writetable( T0, fullfile( path2, filename ) );

end

% save the dataset plots
genPaperDataPlots( thisData, "Exemplars", name );

% re-save the component plots
genPaperCompPlots( path, "Exemplars", name, 2, nReports, nModels ) 