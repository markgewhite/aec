% Run the analysis for the real data sets

clear;

runAnalysis = false;

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
    for i = 2:nReports
    
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
                setup.data.args.SessionType = 'Initial';
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
        %thisData{i} = thisRun.getDatasets( which = "First", set = "Testing" );
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
fields = [ "ReconLossRegular", ...
           "AuxModelLoss" ];

% reshape the results
m = 0;
for i = 1:nReports
    nParams = cellfun( @length, results{i}.GridSearch );
    for j = 1:nParams(2)
        for k = 1:nParams(3)
            m = m + 1;
            resultsExt{m}.BaselineSetup = results{i}.BaselineSetup; %#ok<SAGROW> 
            resultsExt{m}.GridSearch = results{i}.GridSearch; %#ok<SAGROW> 
            flds = fieldnames( results{i}.TestingResults );
            for f = 1:length(flds)
                resultsExt{m}.TestingResults.(flds{f}) = ...
                    results{i}.TestingResults.(flds{f})(:,j,k);
            end
        end
    end
end

groupSizes = [3 3];

T0 = genPaperResultsTable( resultsExt, fields, groupSizes );


TestNames = [ repelem( "GaitRec (GRF)", prod(nParams(2:3)), 1 ); ...
             repelem( "GaitRec (COP)", prod(nParams(2:3)), 1 ) ];
Param1Names = [ "1D"; "1D"; "3D"; "3D"; ...
                "Deriv"; "Deriv"; "-"; "-" ];
Param2Names = repmat( ["Controls vs Disorders"; "Disorders"], prod(nParams(2:3)), 1 );

T0 = addvars( T0, TestNames, Before = 1 );
T0 = addvars( T0, Param1Names, Before = 2 );
T0 = addvars( T0, Param2Names, Before = 3 );

filename = strcat( "RealDatasets-Results.csv" );
writetable( T0, fullfile( path2, filename ) );

