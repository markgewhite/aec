% Run the analysis of the response to key parameter values

clear;

runAnalysis = false;
inParallel = true;
resume = false;
generatePlots = false;
reportIdx = 1:9;
plotDim = [2 5];
maxCoeff = 3;

% set the destinations for results and figures
path0 = fileparts( which('code/parameterAnalysis.m') );
path = [path0 '/../results/params/'];
pathResults = [path0 '/../paper/results/'];

% -- data setup --
setup.data.class = @UCRDataset;
datasets = [ 11, 17, 19, 67, 80, 84, 85, 92, 104, 115 ];
datasetNames = [ "Computers", ...
                 "DistalPhalanxOutlineCorrect", ...
                 "Earthquakes", ...
                 "Strawberry", ...
                 "Wafer", ...
                 "WormsTwoClass", ...
                 "Yoga", ...
                 "FreezerRegularTrain", ...
                 "PowerCons", ...
                 "SemgHandGenderCh2" ];

datasetsVarLen = [ 87, 88, 89, 99, 103, 105, 109, 110 ];
datasetVarLenNames = [  "AllGestureWiimoteX", ...
                        "AllGestureWiimoteY", ...
                        "AllGestureWiimoteZ", ...
                        "PickupGestureWiimoteZ", ...
                        "PLAID", ...
                        "ShakeGestureWiimoteZ", ...
                        "GesturePebbleZ1", ...
                        "GesturePebbleZ2" ];

modelClasses = {@PCAModel, @FCModel, @ConvolutionalModel};
legendNames = [ "PCA", "FC", "Conv" ];

% -- model setup --
setup.model.args.ZDim = 4;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.HasCentredDecoder = true;
setup.model.args.RandomSeed = 1234;
setup.model.args.ShowPlots = true;

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
setup.model.args.trainer.NumIterations = 1000;
setup.model.args.trainer.BatchSize = 5000;
setup.model.args.trainer.UpdateFreq = 2000;
setup.model.args.trainer.Holdout = 0.2;
setup.model.args.trainer.ValType = 'Both';
setup.model.args.trainer.ValFreq = 10;
setup.model.args.trainer.ValPatience = 20;

% evaluations
setup.eval.args.verbose = true;
setup.eval.args.CVType = 'Holdout';
setup.eval.args.KFolds = 2;
setup.eval.args.KFoldRepeats = 5;

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
thisData = cell( nReports, 1 );
memorySaving = 4;

if runAnalysis

    if inParallel
        delete( gcp('nocreate') );
        pool = parpool;
    end

    for i = reportIdx
    
        switch i
            case 1 % ZDim
                parameters = [ "model.class", ...
                               "model.args.ZDim", ...
                               "data.args.SetID" ];

                values = {modelClasses, ...
                          [2 3 4 6 8 10 15 20], ...
                          datasets };

                setup.eval.args.CVType = 'Holdout';

            case 2 % ResampleRate
                parameters = [ "model.class", ...
                               "data.args.ResampleRate", ...
                               "data.args.SetID" ];

                values = {modelClasses, ...
                          [1 5/4 3/2 2 4 5], ...
                          datasets };

                setup.eval.args.CVType = 'Holdout';

            case 3 % NormalizedPts
                parameters = [ "model.class", ...
                               "data.args.NormalizedPts", ...
                               "data.args.SetID" ];

                values = {modelClasses, ...
                          [25 50 75 100 150 200], ...
                          datasets };

                setup.eval.args.CVType = 'Holdout';

            case 4 % HasCentredDecoder
                parameters = [ "model.class", ...
                               "model.args.HasCentredDecoder", ...
                               "data.args.SetID" ];

                values = {modelClasses, ...
                          [false true], ...
                          datasets };

                setup.eval.args.CVType = 'KFold';

            case 5 % HasAdaptiveTimeSpan
                parameters = [ "model.class", ...
                               "data.args.HasAdaptiveTimeSpan", ...
                               "data.args.SetID" ];

                values = {modelClasses, ...
                          [false true], ...
                          datasets };

                setup.eval.args.CVType = 'KFold';

            case 6 % Normalization
                parameters = [ "model.class", ...
                               "model.args.HasInputNormalization", ...
                               "data.args.SetID" ];

                values = {{@ConvolutionalModel}, ...
                          [false true], ...
                          datasetsVarLen };

                setup.eval.args.CVType = 'KFold';

            case 7 % Classifier Loss
                parameters = [ "model.class", ...
                               "model.args.lossFcns.zcls.args.DoCalcLoss", ...
                               "data.args.SetID" ];

                values = {modelClasses, ...
                          [false true], ...
                          datasets };

                setup.eval.args.CVType = 'KFold';

            case 8 % KL Divergence Loss
                parameters = [ "model.class", ...
                               "model.args.lossFcns.kl.args.DoCalcLoss", ...
                               "data.args.SetID" ];

                values = {modelClasses, ...
                          [false true], ...
                          datasets };

                setup.eval.args.CVType = 'KFold';

            case 9 % Adversarial Loss
                parameters = [ "model.class", ...
                               "model.args.lossFcns.adv.args.DoCalcLoss", ...
                               "data.args.SetID" ];

                values = {modelClasses, ...
                          [false true], ...
                          datasets };

                setup.eval.args.CVType = 'KFold';

            otherwise
                error(['Undefined grid search for i = ' num2str(i)]);

        end
    
        if inParallel
            results(i) = parfeval( pool, @investigationResults, 1, ...
                                   names(i), path, ...
                                   parameters, values, setup, ...
                                   memorySaving, resume );
        else
            results(i) = investigationResults( names(i), path, ...
                                               parameters, values, setup, ...
                                               memorySaving, resume );
        end
    
    end

else

    % load from files instead
    for i = reportIdx
        filename = strcat( names(i), "/", names(i), "-Investigation" );
        load( fullfile( path, filename ), 'report' );
        results{i} = report;
    end

    % generate the associated plots
    for i = reportIdx
    
        switch i
            case 1
                plotParam = "Z Dimension";
                plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                               "AuxModelErrorRate", "Aux. Model Error Rate"; ...
                               "AuxModelCoeff", "Aux. Model Coefficients"];
    
            case 2
                plotParam = "X Resampling Rate";
                plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                               "AuxModelErrorRate", "Aux. Model Error Rate"; ...
                               "TotalTime", "Total Execution Time"];
    
            case 3
                plotParam = "$\hat{X}$ Dimension";
                plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                               "AuxModelErrorRate", "Aux. Model Error Rate"; ...
                               "XCCorrelation", "$X_{C}$ Mean Correlation"; ...
                               "TotalTime", "Total Execution Time"];
    
            case 4
                plotParam = "Has Centred Decoder";
                plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                               "AuxModelErrorRate", "Aux. Model Error Rate"; ...
                               "ReconBias", "Reconstruction Loss Bias"];
    
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
                               "AuxModelErrorRate", "Aux. Model Error Rate"; ...
                               "AuxNetworkErrorRate", "Aux. Network Error Rate"];
    
            case 8
                plotParam = "Has KL Loss";
                plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                               "AuxModelErrorRate", "Aux. Model Error Rate"; ...
                               "AuxNetworkErrorRate", "Aux. Network Error Rate"];
    
            case 9
                plotParam = "Has Adversarial Loss";
                plotMetrics = ["ReconLossRegular", "Reconstruction Loss"; ...
                               "AuxModelErrorRate", "Aux. Model Error Rate"; ...
                               "AuxNetworkErrorRate", "Aux. Network Error Rate"];
    
            otherwise
                error(['Undefined parameters for report = ' num2str(i)]);
    
        end

        if i==6
            thisLegend = 'Conv';
        else
            thisLegend = legendNames;
        end

        for j = 1:size(plotMetrics,1)

            switch plotMetrics(j,1)
                case {'AuxModelCoeff', 'TotalTime'}
                    resultSet = 'TrainingResults';
                otherwise
                    resultSet = 'TestingResults';
            end
            x = results{i}.GridSearch{2};
            y = results{i}.(resultSet).Mean.(plotMetrics(j,1));
            ySD = results{i}.(resultSet).SD.(plotMetrics(j,1));

            if iscell(y)
                [y, thisLegend] = extractCellResults( y, maxCoeff, thisLegend );
                ySD = zeros( size(y) );
            else
                y = abs(y);
            end

            if strcmp( plotMetrics(j,1), 'AuxModelCoeff' )
                best = 'Largest';
            else
                best = 'Smallest';
            end

            if strcmp( plotMetrics(j,1), 'TotalTime' )
                y = y/60;
            end

            tbl = paramRelationTable(  x, y, ...
                                       thisLegend, ...
                                       Highlight = best );
            filename = strcat( names(i), "-", plotMetrics(j,1), ".csv" );
            writetable( tbl, fullfile(pathResults, filename), ...
                        WriteRowNames = true );
            
            if generatePlots
                fig = plotParamRelation(   x, y, ySD, ...
                                           plotParam, plotMetrics(j,2), ...
                                           thisLegend, ...
                                           datasetNames, ...
                                           subPlotDim = plotDim, ...
                                           squarePlot = true ); 
    
                filename = strcat( names(i), "-", plotMetrics(j,1), ".pdf" );
                fig = formatIEEEFig( fig, ...
                                     width = "Page", ...
                                     height = 2.00, ...
                                     keepYAxisTicks = true, ...
                                     keepTitle = false, ...
                                     filename = filename );
            end
            
        end

    end

end


