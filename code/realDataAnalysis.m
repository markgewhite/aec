% Run the analysis for the real data sets

clear;

runAnalysis = true;

% set the destinations for results and figures
path = fileparts( which('code/realDataAnalysis.m') );
path = [path '/../results/real/'];

path2 = fileparts( which('code/realDataAnalysis.m') );
path2 = [path2 '/../paper/results/'];

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
setup.model.args.ZDim = 4;
setup.model.args.InitZDimActive = 0;
setup.model.args.KFolds = 1;
setup.model.args.AuxModelType = 'SVM';
setup.model.args.randomSeed = 1234;

% -- trainer setup --
setup.model.args.trainer.holdout = 0.2;

% -- grid search --
nDims = 4;
nModels = 3;
nReports = 2;

name = [ "GaitRecGRF", ...
         "GaitRecCOP" ];
results = cell( nReports, 1 );
thisData = cell( nReports, 1 );
memorySaving = 4;

if runAnalysis
    for i = 1:nReports
    
        switch i

            case 1
                % GaitRec dataset using GRF data
                setup.data.class = @GaitrecDataset;
                setup.data.args.HasGRF = true;
                setup.data.args.HasCOP = false;
                setup.data.args.SessionType = 'Initial';
                setup.data.args.FromMatlabFile = false;
                setup.data.args.OverSmoothing = 1E3;
                
                parameters = [ "model.class", ...
                               "data.args.HasVGRFOnly", ...
                               "data.args.Grouping" ];
                values = { {@FCModel, @ConvolutionalModel, @FullPCAModel}, ...
                           {true,false}, ...
                           {'ControlsVsDisorders', 'Disorders'} }; 

                setup.model.args.trainer.numEpochs = 100;
                setup.model.args.trainer.updateFreq = 25;
                setup.model.args.trainer.numEpochsPreTrn = 5;

                setup.model.args.trainer.batchSize = 250;

            case 2
                % GaitRec dataset using COP data
                setup.data.class = @GaitrecDataset;
                setup.data.args.HasGRF = false;
                setup.data.args.HasVGRFOnly = false;
                setup.data.args.HasCOP = true;
                setup.data.args.SessionType = 'Control';
                setup.data.args.FromMatlabFile = false;
                
                parameters = [ "model.class", ...
                               "data.args.HasDerivative", ...
                               "data.args.Grouping" ];
                values = { {@FCModel, @ConvolutionalModel, @FullPCAModel}, ...
                           {true,false}, ...
                           {'ControlsVsDisorders', 'Disorders'} }; 

                setup.model.args.trainer.numEpochs = 100;
                setup.model.args.trainer.updateFreq = 25;
                setup.model.args.trainer.numEpochsPreTrn = 5;

                setup.model.args.trainer.batchSize = 250; 

        end
    
        thisRun = Investigation( name(i), path, ...
                                 parameters, values, setup, memorySaving );
        argsCell = namedargs2cell( setup.data.args );
        thisData{i} = thisRun.getDatasets( which = "First", set = "Testing" );
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
                           groups = [ {1:nModels}, {3*nModels+1:4*nModels} ] );

    filename = strcat( "Synthetic-Dim", num2str(d), ".csv" );
    writetable( T0, fullfile( path2, filename ) );

end

