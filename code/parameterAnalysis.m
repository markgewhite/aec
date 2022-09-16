% Run the analysis of the response to key parameter values

clear;

runAnalysis = true;

% set the destinations for results and figures
path0 = fileparts( which('code/parameterAnalysis.m') );
path = [path0 '/../results/params/'];
pathResults = [path0 '/../paper/results/'];

% -- data setup --
setup.data.class = @UCRDataset;
datasets = [17 31]; %[ 17, 31, 38, 74, 104 ];

% -- model setup --
setup.model.args.ZDim = 4;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.HasCentredDecoder = true;
setup.model.args.RandomSeed = 1234;

% -- loss functions --
setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
setup.model.args.lossFcns.recon.name = 'Reconstruction';
setup.model.args.lossFcns.zcls.class = @ClassifierLoss;
setup.model.args.lossFcns.zcls.name = 'ZClassifier';
setup.model.args.lossFcns.adv.class = @AdversarialLoss;
setup.model.args.lossFcns.adv.name = 'Discriminator';
setup.model.args.lossFcns.adv.args.DoCalcLoss = false;
setup.model.args.lossFcns.kl.class = @KLDivergenceLoss;
setup.model.args.lossFcns.kl.name = 'KLDivergence';
setup.model.args.lossFcns.kl.args.DoCalcLoss = false;

% -- trainer setup --
setup.model.args.trainer.UpdateFreq = 200;
setup.model.args.trainer.BatchSize = 100;
setup.model.args.trainer.Holdout = 0.2;
setup.model.args.trainer.ValType = 'Both';
setup.model.args.trainer.ValFreq = 5;
setup.model.args.trainer.ValPatience = 40;

% evaluations
setup.eval.args.verbose = true;
setup.eval.args.CVType = 'KFold';
setup.eval.args.KFolds = 2;
setup.eval.args.KFoldRepeats = 1;

names = [ "ZDimRelation", ...
          "XInputDimRelation", ...
          "XTargetDimRelation", ...
          "HasCentredDecoder", ...
          "HasAdaptiveTimeSpan" ];
nReports = length( names );
results = cell( nReports, 1 );
thisData = cell( nReports, 1 );
memorySaving = 4;

if runAnalysis
    for i = 1:nReports
    
        switch i
            case 1 % ZDim
                parameters = [ "model.class", ...
                               "model.args.ZDim", ...
                               "data.args.SetID" ];
                 % [2 3 4 6 8 10 15 20], ...

                values = {{@FCModel, @PCAModel, @FCModel, @ConvolutionalModel}, ...
                          [2 3], ...
                          datasets };
            otherwise
                error(['Undefined grid search for i = ' num2str(i)]);
        end
    
        thisRun = Investigation( names(i), path, ...
                                 parameters, values, setup, memorySaving );
        results{i} = thisRun.getResults;
        clear thisRun;
    
    end

else

    % load from files instead
    for i = 1:nReports
        filename = strcat( names(i), "/", names(i), "-Investigation" );
        load( fullfile( path, filename ), 'report' );
        results{i} = report;
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
    
T0 = genPaperResultsTable( results, fields, groupSizes );

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


