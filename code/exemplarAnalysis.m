% Run the analysis for the exemplar data sets

clear;

% set the results destination
path = fileparts( which('code/exemplarAnalysis.m') );
path = [path '/../results/exemplars/'];

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

nModels = length( values{1} );
nDatasets = 6;
nReports = 8;

name = [ "1G-PeakVar", "1G-MeanVar", ...
         "2G-PeakVar", "2G-MeanSDVar", ...
         "1G-2Classes", "1G-2Classes-Classify", ...
         "2G-2Classes", "2G-2Classes-Classify" ];
results = cell( nReports, 1 );

k = 0;
for i = 1:nDatasets

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
    
    end

    switch i

        case 1
            % Single Gaussian with peak (height) variance   
            setup.data.args.PeakCovariance{1} = 1;
            setup.data.args.MeanCovariance{1} = 1E-6;
            setup.data.args.SDCovariance{1} = 1E-6;
       
            k = k + 1;
            setup.lossFcns = lossFcns1;
            thisRun = Investigation( name(k), path, parameters, values, setup );
            results{k} = thisRun.getResults;

        case 2
            % Single Gaussian with mean (position) variance
            setup.data.args.PeakCovariance{1} = 1E-6;
            setup.data.args.MeanCovariance{1} = 1;
            setup.data.args.SDCovariance{1} = 1E-6;
    
            k = k + 1;
            setup.lossFcns = lossFcns1;
            thisRun = Investigation( name(k), path, parameters, values, setup );
            results{k} = thisRun.getResults;

        case 3
            % Double Gaussian with peak inverse covariance
            setup.data.args.PeakCovariance{1} = [1 -sigma; -sigma 1];
            setup.data.args.MeanCovariance{1} = 1E-6*eye(2);
            setup.data.args.SDCovariance{1} = 1E-6*eye(2);
       
            k = k + 1;
            setup.lossFcns = lossFcns1;
            thisRun = Investigation( name(k), path, parameters, values, setup );
            results{k} = thisRun.getResults;

        case 4
            % Double Gaussian with peak inverse covariance   
            setup.data.args.PeakCovariance{1} = [1 sigma; sigma 1];
            setup.data.args.MeanCovariance{1} = [1 -sigma; -sigma 1];
            setup.data.args.SDCovariance{1} = [1 sigma; sigma 1];
       
            k = k + 1;
            setup.lossFcns = lossFcns1;
            thisRun = Investigation( name(k), path, parameters, values, setup );
            results{k} = thisRun.getResults;

        case 5
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
       
            k = k + 1;
            setup.lossFcns = lossFcns1;
            thisRun = Investigation( name(k), path, parameters, values, setup );
            results{k} = thisRun.getResults;

            k = k + 1;
            setup.lossFcns = lossFcns2;
            thisRun = Investigation( name(k), path, parameters, values, setup );
            results{k} = thisRun.getResults;

        case 6
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
       
            k = k + 1;
            setup.lossFcns = lossFcns1;
            thisRun = Investigation( name(k), path, parameters, values, setup );
            results{k} = thisRun.getResults;

            k = k + 1;
            setup.lossFcns = lossFcns2;
            thisRun = Investigation( name(k), path, parameters, values, setup );
            results{k} = thisRun.getResults;            

    end

end

% temporary file read
name = [ "1G-PeakVar", "1G-MeanVar", ...
         "2G-PeakVar", "2G-MeanSDVar", ...
         "1G-2Classes", "1G-2Classes-Classify", ...
         "2G-2Classes", "2G-2Classes-Classify" ];
results = cell( 8, 1 );

for k = 1:8
    filename = strcat( name(k), "/", name(k), "-Investigation" );
    load( fullfile( path, filename ), 'report' );
    results{k} = report;
end

% compile results for the paper
fields = [ "ReconLossRegular", "AuxModelLoss", ...
           "ZCorrelation", "XCCorrelation" ];
nFields = length( fields );
for d = 1:nDims

    for i = 1:nFields
        for j = 1:nModels
            fieldName = strcat( fields(i), num2str(j) );
            T.(fieldName) = zeros( nDatasets, 1 );
            for k = 1:nDatasets
                T.(fieldName)(k) = ...
                    round( results{k}.TestingResults.(fields(i))(j,d), 3 );
            end
        end
    end
    T0 = struct2table( T );
    filename = strcat( "Results-Dim", num2str(d), ".csv" );
    writetable( T0, fullfile( path, filename ) );
end
