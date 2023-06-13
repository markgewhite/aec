% Run the models to evaluate their performance across multiple data sets

clear;
close all;

runAnalysis = true;
catchErrors = false;

reportIdx = 5;

plotDim = 4;
maxCoeff = 3;

rng('default');

% set the destinations for results and figures
path0 = fileparts( which('code/performanceAnalysis.m') );
path = [path0 '/../results/test/'];
pathResults = [path0 '/../paper/results/'];

% -- model setup --
setup.model.class = @ConvBranchedModel;
setup.model.args.IsVAE = true;
setup.model.args.UseEncodingMean = false;
setup.model.args.NumEncodingDraws = 1;
setup.model.args.ZDim = 2;
setup.model.args.NumHidden = 3;
setup.model.args.FilterSize = 4;
setup.model.args.Stride = 2;
setup.model.args.Pooling = 'None';
setup.model.args.NumHiddenDecoder = 4;
setup.model.args.NumFiltersDecoder = 8;
setup.model.args.FilterSizeDecoder = 3;
setup.model.args.StrideDecoder = 3;
setup.model.args.InputDropout = 0;
setup.model.args.Dropout = 0;
setup.model.args.NetNormalizationType = 'None';
setup.model.args.NetActivationType = 'None';

%setup.model.args.NumFCDecoder = 10;
%setup.model.args.FCFactorDecoder = 0;
%setup.model.args.NetNormalizationTypeDecoder = 'None';
%setup.model.args.NetActivationTypeDecoder = 'Relu';

setup.model.args.ComponentType = 'AEC';
setup.model.args.NumCompLines = 7;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.AuxObjective = 'Classification';
setup.model.args.randomSeed = 1234;
setup.model.args.HasCentredDecoder = true;
setup.model.args.ShowPlots = true;

% -- loss functions --
setup.model.args.lossFcns.recon.class = @ReconstructionLoss;
setup.model.args.lossFcns.recon.name = 'Reconstruction';

setup.model.args.lossFcns.reconrough.class = @ReconstructionRoughnessLoss;
setup.model.args.lossFcns.reconrough.name = 'ReconstructionRoughness';
setup.model.args.lossFcns.reconrough.args.Dilations = 1;
setup.model.args.lossFcns.reconrough.args.UseLoss = false;

setup.model.args.lossFcns.kl.class = @KLDivergenceLoss;
setup.model.args.lossFcns.kl.name = 'KLDivergence';
setup.model.args.lossFcns.kl.args.Beta = 1E-2;
setup.model.args.lossFcns.kl.args.UseLoss = true;

setup.model.args.lossFcns.zorth.class = @OrthogonalLoss;
setup.model.args.lossFcns.zorth.name = 'ZOrthogonality';
setup.model.args.lossFcns.zorth.args.UseLoss = true;

setup.model.args.lossFcns.xorth.class = @ComponentLoss;
setup.model.args.lossFcns.xorth.name = 'XOrthogonality';
setup.model.args.lossFcns.xorth.args.Criterion = 'Orthogonality';
setup.model.args.lossFcns.xorth.args.Alpha = 1E-1;
setup.model.args.lossFcns.xorth.args.UseLoss = true;

setup.model.args.lossFcns.xvar.class = @ComponentLoss;
setup.model.args.lossFcns.xvar.name = 'XVarimax';
setup.model.args.lossFcns.xvar.args.Criterion = 'Varimax';
setup.model.args.lossFcns.xvar.args.UseLoss = false;

setup.model.args.lossFcns.zcls.class = @ClassifierLoss;
setup.model.args.lossFcns.zcls.name = 'ZClassifier';
setup.model.args.lossFcns.zcls.args.NumHidden = 1;
setup.model.args.lossFcns.zcls.args.NumFC= 10;
setup.model.args.lossFcns.zcls.args.HasBatchNormalization = false;
setup.model.args.lossFcns.zcls.args.ReluScale = 0;
setup.model.args.lossFcns.zcls.args.Dropout = 0;

%setup.model.args.lossFcns.zreg.class = @RegressionLoss;
%setup.model.args.lossFcns.zreg.name = 'ZRegressor';
%setup.model.args.lossFcns.zreg.args.NumHidden = 1;
%setup.model.args.lossFcns.zreg.args.NumFC= 10;
%setup.model.args.lossFcns.zreg.args.HasBatchNormalization = false;
%setup.model.args.lossFcns.zreg.args.ReluScale = 0;
%setup.model.args.lossFcns.zreg.args.Dropout = 0;
%setup.model.args.lossFcns.zreg.args.UseLoss = false;

% -- trainer setup --
setup.model.args.trainer.NumIterations = 1000;
setup.model.args.trainer.UpdateFreq = 2000;
setup.model.args.trainer.Holdout = 0;
setup.model.args.trainer.ShowPlots = false;

% --- evaluation setup ---
setup.eval.args.CVType = 'Holdout';
setup.eval.args.KFolds = 2;
setup.eval.args.KFoldRepeats = 5;
setup.eval.args.InParallel = false;

% --- investigation setup ---
models = {@ConvBranchedModel};

dims = [2];
compTypes = {'PDP', 'AEC'};
parameters = [ "model.args.NumFilters", ...
               "model.args.FilterSize", ...
               "model.args.Stride", ...
               "model.args.NumFiltersDecoder", ...
               "model.args.FilterSizeDecoder", ...
               "model.args.StrideDecoder", ...
               "model.args.lossFcns.zcls.args.UseLoss" ];
values = {  [8 16], ...
            [3 4 5], ...
            [2 3], ...
            [8 16], ...
            [3 4 5], ...
            [2 3], ...
            {false true}}; 
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
                setup.data.args.NormalizedPts = 21;
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
                setup.data.args.NormalizedPts = 21;
                setup.model.args.trainer.BatchSize = 3000;

            case 3
                % Fukuchi ground reaction force in three dimensions
                name = 'Fukuchi-GRF-3D';
                setup.data.class = @FukuchiDataset;
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.YReference = 'AgeGroup';
                setup.data.args.Category = 'Ground';
                setup.data.args.NormalizedPts = 21;
                setup.model.args.trainer.BatchSize = 1000;

            case 4
                % Fukuchi hip, knee and ankle joint angles
                name = 'Fukuchi-JointAngles-3D';
                setup.data.class = @FukuchiDataset;
                setup.data.args.HasGRF = true;
                setup.data.args.HasVGRFOnly = false;
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.NormalizedPts = 21;
                setup.data.args.YReference = 'AgeGroup';
                setup.data.args.Category = 'JointAngles';
                setup.data.args.HasHipAngles = true;
                setup.data.args.HasKneeAngles = true;
                setup.data.args.HasAnkleAngles = true;
                setup.data.args.SagittalPlaneOnly = true;
                setup.model.args.trainer.BatchSize = 75;

            case 5
                % Jumps vertical ground reaction force
                name = 'JumpGRF';
                setup.data.class = @JumpGRFDataset;
                setup.data.args.OutcomeVar = 'JumpType';
                setup.data.args.Normalization = 'PAD';
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.NormalizedPts = 21;
                setup.data.args.ResampleRate = 5;
                setup.model.args.trainer.BatchSize = 75;

            case 6
                % Fukuchi ground reaction force in one dimension
                name = 'Fukuchi-GRF-1D';
                setup.data.class = @FukuchiDataset;
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.NormalizedPts = 21;
                setup.data.args.HasGRF = true;
                setup.data.args.HasVGRFOnly = true;
                setup.data.args.YReference = 'AgeGroup';
                setup.data.args.Category = 'Ground';
                setup.model.args.trainer.BatchSize = 1000;

            case 7
                % Fukuchi hip, knee and ankle joint angles
                name = 'Fukuchi-JointAngles-1D';
                setup.data.class = @FukuchiDataset;
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.NormalizedPts = 21;
                setup.data.args.YReference = 'AgeGroup';
                setup.data.args.Category = 'JointAngles';
                setup.data.args.HasHipAngles = false;
                setup.data.args.HasKneeAngles = true;
                setup.data.args.HasAnkleAngles = false;
                setup.data.args.SagittalPlaneOnly = true;
                setup.model.args.trainer.BatchSize = 75;

            otherwise
                % Synthetic data set
                setup.data.class = @SyntheticDataset;
                setup.data.args.TemplateSeed = randi(1E6);
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.NormalizedPts = 21;
                setup.data.args.ClassSizes = [ 250, 250 ];
                setup.model.args.trainer.BatchSize = 75;

                name = ['Synthetic-' num2str(setup.data.args.TemplateSeed)];

        end
           
        myInvestigations{i} = ParallelInvestigation( name, path, parameters, values, ...
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


