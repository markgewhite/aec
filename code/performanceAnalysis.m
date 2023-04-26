% Run the models to evaluate their performance across multiple data sets

clear;

runAnalysis = true;
catchErrors = false;

reportIdx = [4];

plotDim = [2 5];
maxCoeff = 3;

rng('default');

% set the destinations for results and figures
path0 = fileparts( which('code/performanceAnalysis.m') );
path = [path0 '/../results/perf/'];
pathResults = [path0 '/../paper/results/'];

% -- data setup --
setup.data.args.normalizedPts = 21;

% -- model setup --
setup.model.args.NumHidden = 1;
setup.model.args.NumFC = 50;
setup.model.args.InputDropout = 0;
setup.model.args.Dropout = 0.1;
setup.model.args.NetNormalizationType = 'Batch';
setup.model.args.NetActivationType = 'Relu';

setup.model.args.NumHiddenDecoder = 2;
setup.model.args.NumFCDecoder = 50;
setup.model.args.FCFactorDecoder = 0;
setup.model.args.NetNormalizationTypeDecoder = 'None';
setup.model.args.NetActivationTypeDecoder = 'None';

setup.model.args.AuxModel = 'Logistic';
setup.model.args.randomSeed = 1234;
setup.model.args.HasCentredDecoder = true;
setup.model.args.ShowPlots = true;

% -- loss functions --
setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
setup.model.args.lossFcns.recon.name = 'Reconstruction';

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
setup.model.args.trainer.UpdateFreq = 5000;
setup.model.args.trainer.Holdout = 0;

% --- evaluation setup ---
setup.eval.args.CVType = 'KFold';
setup.eval.args.KFolds = 2;
setup.eval.args.KFoldRepeats = 5;
setup.eval.args.InParallel = false;

% --- investigation setup ---
models = {@PCAModel, @BranchedFCModel};

dims = [2 3 4 5];
parameters = [ "model.class", ...
               "model.args.ZDim"];
values = {models, dims}; 
memorySaving = 3;
myInvestigations = cell( length(reportIdx), 1 );

if runAnalysis

    for i = reportIdx

        if isfield(setup, 'data')
            % reset the data settings
            setup = rmfield( setup, 'data' );
        end

        % select the data set
        switch i

            case 1
                % Gaitrec ground reaction force in one dimension
                name = 'Gaitrec-GRF-1D';
                setup.data.class = @GaitrecDataset;
                setup.data.args.HasGRF = true;
                setup.data.args.HasVGRFOnly = true;
                setup.data.args.Grouping = 'ControlsVsDisorders';
                setup.data.args.ShodCondition = 'Barefoot/Socks';
                setup.data.args.Speed = 'SelfSelected';
                setup.data.args.SessionType = 'All';
                setup.data.args.Side = 'Affected';
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.NormalizedPts = 5;
                setup.model.args.trainer.BatchSize = 3000;

            case 2
                % Gaitrec ground reaction force in three dimensions
                name = 'Gaitrec-GRF-3D';
                setup.data.class = @GaitrecDataset;
                setup.data.args.HasGRF = true;
                setup.data.args.HasVGRFOnly = false;
                setup.data.args.Grouping = 'ControlsVsDisorders';
                setup.data.args.ShodCondition = 'Barefoot/Socks';
                setup.data.args.Speed = 'SelfSelected';
                setup.data.args.SessionType = 'All';
                setup.data.args.Side = 'Affected';
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.NormalizedPts = 5;
                setup.model.args.trainer.BatchSize = 3000;

            case 3
                % Fukuchi ground reaction force in three dimensions
                name = 'Fukuchi-GRF-3D';
                setup.data.class = @FukuchiDataset;
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.YReference = 'AgeGroup';
                setup.data.args.Category = 'Ground';
                setup.data.args.NormalizedPts = 5;
                setup.model.args.trainer.BatchSize = 1000;

            case 4
                % Fukuchi hip, knee and ankle joint angles
                name = 'Fukuchi-JointAngles-3D';
                setup.data.class = @FukuchiDataset;
                setup.data.args.HasGRF = true;
                setup.data.args.HasVGRFOnly = false;
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.YReference = 'AgeGroup';
                setup.data.args.Category = 'JointAngles';
                setup.data.args.HasHipAngles = true;
                setup.data.args.HasKneeAngles = true;
                setup.data.args.HasAnkleAngles = true;
                setup.data.args.SagittalPlaneOnly = true;
                setup.data.args.NormalizedPts = 5;
                setup.model.args.trainer.BatchSize = 75;

            case 5
                % Jumps vertical ground reaction force
                name = 'JumpGRF';
                setup.data.class = @JumpGRFDataset;
                setup.data.args.Normalization = 'PAD';
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.ResampleRate = 5;
                setup.data.args.NormalizedPts = 5;
                setup.model.args.trainer.BatchSize = 75;

            otherwise
                % Synthetic data set
                setup.data.class = @SyntheticDataset;
                setup.data.args.TemplateSeed = randi(1E6);
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.ClassSizes = [ 250, 250 ];
                setup.model.args.trainer.BatchSize = 75;

                name = ['Synthetic-' num2str(setup.data.args.TemplateSeed)];

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


