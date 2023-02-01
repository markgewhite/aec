% Run the analysis of the response to key parameter values

clear;

runAnalysis = true;
inParallel = true;
resume = false;
catchErrors = false;
reportIdx = 1:3;
plotDim = [2 5];
maxCoeff = 3;

% set the destinations for results and figures
path0 = fileparts( which('code/parameterAnalysis.m') );
path = [path0 '/../results/params/'];
pathResults = [path0 '/../paper/results/'];

% -- model setup --
setup.model.class = @AsymmetricFCModel;

setup.model.args.FCFactor = 2;
setup.model.args.InputDropout = 0.2;
setup.model.args.ReLuScale = 0.2;
setup.model.args.HasBatchNormalization = true;
setup.model.args.Dropout = 0;

setup.model.args.FCFactorDecoder = setup.model.args.FCFactor;
setup.model.args.ReLuScaleDecoder = setup.model.args.ReLuScale;
setup.model.args.HasBatchNormalizationDecoder = setup.model.args.HasBatchNormalization;
setup.model.args.DropoutDecoder = setup.model.args.Dropout;

setup.model.args.AuxModel = 'Logistic';
setup.model.args.ComponentType = 'PDP';
setup.model.args.HasCentredDecoder = true;
setup.model.args.RandomSeed = 1234;
setup.model.args.ShowPlots = false;

% -- loss functions --
setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
setup.model.args.lossFcns.recon.name = 'Reconstruction';
setup.model.args.lossFcns.zcls.class = @ClassifierLoss;
setup.model.args.lossFcns.zcls.name = 'ZClassifier';

% -- trainer setup --
setup.model.args.trainer.NumIterations = 5000;
setup.model.args.trainer.BatchSize = 100;
setup.model.args.trainer.UpdateFreq = 10000;
setup.model.args.trainer.Holdout = 0.2;

% --- evaluation setup ---
setup.eval.args.CVType = 'KFold';
setup.eval.args.KFolds = 2;
setup.eval.args.KFoldRepeats = 5;

% --- investigation setup ---
parameters = [ "model.args.ZDim", ...
               "data.args.NormalizedPts" ];
values = {[2 3 4 6 8], ...
          [10 20 30 50 100]}; 

names = [ "JumpsVGRF", ...
          "GaitrecGRF", ...
          "FukuchiJointAngles" ];
nReports = length( names );
thisData = cell( nReports, 1 );
memorySaving = 3;

if runAnalysis

    if inParallel
        delete( gcp('nocreate') );
        pool = parpool;
    end

    for i = reportIdx

        if isfield(setup, 'data')
            % reset the data settings
            setup = rmfield( setup, 'data' );
        end

        % select the data set
        switch i

            case 1
                % Jumps vertical ground reaction force
                setup.data.class = @JumpGRFDataset;
                setup.data.args.Normalization = 'PAD';
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.ResampleRate = 5;

            case 2
                % Gaitrec ground reaction force
                setup.data.class = @GaitrecDataset;
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.MaxObs = 1000;
                setup.data.args.Grouping = 'ControlsVsDisorders';
                setup.data.args.ShodCondition = 'Barefoot/Socks';
                setup.data.args.Speed = 'SelfSelected';
                setup.data.args.SessionType = 'All';
                setup.data.args.Side = 'Affected';

            case 3
                % Fukuchi hip, knee and ankle joint angles
                setup.data.class = @FukuchiDataset;
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.YReference = 'AgeGroup';
                setup.data.args.Category = 'JointAngles';
                setup.data.args.HasHipAngles = true;
                setup.data.args.HasKneeAngles = true;
                setup.data.args.HasAnkleAngles = true;
                setup.data.args.SagittalPlaneOnly = true;
            
            otherwise
                error('Unrecognised dataset ID.');

        end
    
        for j = 1:4
            % Set the autoencoder design
            if j<=2
                % Low-complexity encoder
                setup.model.args.NumHidden = 2;
                setup.model.args.NumFC = 64;
            else
                % High-complexity encoder
                setup.model.args.NumHidden = 3;
                setup.model.args.NumFC = 1024;
            end
            if mod(j,2)==1
                % Low-complexity decoder
                setup.model.args.NumHiddenDecoder = 2;
                setup.model.args.NumFCDecoder = 64;
            else
                % High-complexity decoder
                setup.model.args.NumHiddenDecoder = 3;
                setup.model.args.NumFCDecoder = 1024;
            end
        
            if inParallel
                results(i,j) = parfeval( pool, @investigationResults, 1, ...
                                       names(i), path, ...
                                       parameters, values, setup, ...
                                       resume, catchErrors, memorySaving );
            else
                results(i,j) = investigationResults( names(i), path, ...
                                                   parameters, values, setup, ...
                                                   resume, catchErrors, memorySaving );
            end
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
    legendNames = [ "AE-LL", "AE-LH", "AE-HL", "AE-HH" ];

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
                                       plotParam, ...
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


