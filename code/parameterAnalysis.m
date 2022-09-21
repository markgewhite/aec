% Run the analysis of the response to key parameter values

clear;

runAnalysis = true;
resume = false;
reportIdx = [3 4 5];

% set the destinations for results and figures
path0 = fileparts( which('code/parameterAnalysis.m') );
path = [path0 '/../results/params/'];
pathResults = [path0 '/../paper/results/'];

% -- data setup --
setup.data.class = @UCRDataset;
datasets = [ 17, 31, 38, 74, 104 ];
datasetNames = [ "DistalPhalanx", ...
                 "GunPoint", ...
                 "ItalyPowerDemand", ...
                 "TwoLeadECG", ...
                 "PowerCons" ];

datasetsVarLen = [ 87, 88, 89, 99, 103, 105, 109, 110 ];
datasetVarLenNames = [  "AllGestureWiimoteX", ...
                        "AllGestureWiimoteY", ...
                        "AllGestureWiimoteZ", ...
                        "PickupGestureWiimoteZ", ...
                        "PLAID", ...
                        "ShakeGestureWiimoteZ", ...
                        "GesturePebbleZ1", ...
                        "GesturePebbleZ2" ];

legendNames = [ "PCA", "FC", "Conv" ];

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
setup.model.args.trainer.ValFreq = 10;
setup.model.args.trainer.ValPatience = 20;

% evaluations
setup.eval.args.verbose = true;
setup.eval.args.CVType = 'KFold';
setup.eval.args.KFolds = 2;
setup.eval.args.KFoldRepeats = 2;

names = [ "ZDimRelation", ...
          "ResampleRateRelation", ...
          "NormalizedPts", ...
          "HasCentredDecoder", ...
          "HasAdaptiveTimeSpan", ...
          "HasNormalizedInput", ...
          "HasClassifierLoss", ...
          "HasKLLoss", ...
          "HasAdversarialLoss" ];
nReports = length( names );
results = cell( nReports, 1 );
thisData = cell( nReports, 1 );
memorySaving = 4;

if runAnalysis
    for i = reportIdx
    
        switch i
            case 1 % ZDim
                parameters = [ "model.class", ...
                               "model.args.ZDim", ...
                               "data.args.SetID" ];

                values = {{@PCAModel, @FCModel, @ConvolutionalModel}, ...
                          [2 3 4 6 8 10 15 20], ...
                          datasets };

            case 2 % ResampleRate
                parameters = [ "model.class", ...
                               "data.args.ResampleRate", ...
                               "data.args.SetID" ];

                values = {{@PCAModel, @FCModel, @ConvolutionalModel}, ...
                          [1 5/4 3/2 2 4 5], ...
                          datasets };

            case 3 % NormalizedPts
                parameters = [ "model.class", ...
                               "data.args.NormalizedPts", ...
                               "data.args.SetID" ];

                values = {{@PCAModel, @FCModel, @ConvolutionalModel}, ...
                          [25 50 75 100 150 200], ...
                          datasets };

            case 4 % HasCentredDecoder
                parameters = [ "model.class", ...
                               "model.args.HasCentredDecoder", ...
                               "data.args.SetID" ];

                values = {{@PCAModel, @FCModel, @ConvolutionalModel}, ...
                          [false true], ...
                          datasets };

            case 5 % HasAdaptiveTimeSpan
                parameters = [ "model.class", ...
                               "model.args.HasAdaptiveTimeSpan", ...
                               "data.args.SetID" ];

                values = {{@PCAModel, @FCModel, @ConvolutionalModel}, ...
                          [false true], ...
                          datasets };

            case 6 % Normalization
                parameters = [ "model.class", ...
                               "model.args.HasNormalizedInput", ...
                               "data.args.SetID" ];

                values = {{@ConvolutionalModel}, ...
                          [false true], ...
                          datasetsVarLen };

            case 7 % Classifier Loss
                parameters = [ "model.class", ...
                               "model.args.lossFcns.zcls.args.DoCalcLoss", ...
                               "data.args.SetID" ];

                values = {{@PCAModel, @FCModel, @ConvolutionalModel}, ...
                          [false true], ...
                          datasets };

            case 8 % KL Divergence Loss
                parameters = [ "model.class", ...
                               "model.args.lossFcns.kl.args.DoCalcLoss", ...
                               "data.args.SetID" ];

                values = {{@PCAModel, @FCModel, @ConvolutionalModel}, ...
                          [false true], ...
                          datasets };

            case 9 % Adversarial Loss
                parameters = [ "model.class", ...
                               "model.args.lossFcns.adv.args.DoCalcLoss", ...
                               "data.args.SetID" ];

                values = {{@PCAModel, @FCModel, @ConvolutionalModel}, ...
                          [false true], ...
                          datasets };


            otherwise
                error(['Undefined grid search for i = ' num2str(i)]);

        end
    
        thisRun = Investigation( names(i), path, ...
                                 parameters, values, setup, ...
                                 memorySaving, resume );
        results{i} = thisRun.getResults;
        clear thisRun;
    
    end

else

    % load from files instead
    for i = reportIdx
        filename = strcat( names(i), "/", names(i), "-Investigation" );
        load( fullfile( path, filename ), 'report' );
        results{i} = report;
    end

end

% generate the associated plots
for i = reportIdx

    switch i
        case 1
            plotParam = "Z Dimension";
            plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                           "AuxModelErrorRate", "Aux. Model Error Rate"];

        case 2
            plotParam = "X Resampling Rate";
            plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                           "AuxModelErrorRate", "Aux. Model Error Rate"];

        case 3
            plotParam = "\hat{X} Dimension";
            plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                           "AuxModelErrorRate", "Aux. Model Error Rate"; ...
                           "XCCorrelation", "X_{C} Mean Correlation"];

        case 4
            plotParam = "Has Centred Decoder";
            plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                           "AuxModelErrorRate", "Aux. Model Error Rate"; ...
                           "ReconLossBias", "Reconstruction Loss Bias"];

        case 5
            plotParam = "Has Adaptive Time Span";
            plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                           "AuxModelErrorRate", "Aux. Model Error Rate"];

        case 6
            plotParam = "Has Time-Normalized Input";
            plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                           "AuxModelErrorRate", "Aux. Model Error Rate"];

        case 7
            plotParam = "Has Classifier Loss";
            plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                           "AuxModelErrorRate", "Aux. Model Error Rate", ...
                           "AuxNetworkErrorRate", "Aux. Network Error Rate"];

        case 8
            plotParam = "Has KL Loss";
            plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                           "AuxModelErrorRate", "Aux. Model Error Rate", ...
                           "AuxNetworkErrorRate", "Aux. Network Error Rate"];

        case 9
            plotParam = "Has Adversarial Loss";
            plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                           "AuxModelErrorRate", "Aux. Model Error Rate", ...
                           "AuxNetworkErrorRate", "Aux. Network Error Rate"];

        otherwise
            error(['Undefined parameters for report = ' num2str(i)]);

    end

    for j = 1:size(plotMetrics,1)
        
        fig = plotParamRelation(   results{i}, plotParam, ...
                                   plotMetrics(j,1), plotMetrics(j,2), ...
                                   datasetNames, legendNames ); 

        filename = strcat( names(i), "-", plotMetrics(j,1), ".pdf" );
        fig = formatIEEEFig( fig, ...
                             width = "Page", ...
                             size = "Medium", ...
                             keepYAxisTicks = true, ...
                             keepTitle = true, ...
                             keepLegend = false, ...
                             filename = filename );

    end

end

