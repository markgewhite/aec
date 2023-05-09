% Run the analysis of the response to key parameter values

clear;
close all;

runAnalysis = true;
catchErrors = true;

reportIdx = 1:3;

rng('default');

% set the destinations for results and figures
path0 = fileparts( which('code/parameterAnalysis.m') );
path = [path0 '/../results/paramDropout/'];
pathResults = [path0 '/../paper/results/'];

% -- data setup --
setup.data.args.normalizedPts = 21;

% -- model setup --
setup.model.class = @BranchedFCModel;
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
setup.model.args.ShowPlots = false;

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
setup.model.args.trainer.BatchSize = 75;
setup.model.args.trainer.UpdateFreq = 5000;
setup.model.args.trainer.Holdout = 0;
setup.model.args.trainer.ShowPlots = false;

% --- evaluation setup ---
setup.eval.args.CVType = 'KFold';
setup.eval.args.KFolds = 2;
setup.eval.args.KFoldRepeats = 2;
setup.eval.args.InParallel = true;

% --- investigation setup ---
parameters = [ "model.args.InputDropout", ...
               "model.args.Dropout" ];
values = {[0.0 0.1 0.2 0.3 0.4 0.5], ...
          [0.0 0.1 0.2 0.3 0.4 0.5]}; 

memorySaving = 3;
myInvestigations = cell( length(reportIdx), 1 );

if runAnalysis

    if isfield(setup, 'data')
        % reset the data settings
        setup = rmfield( setup, 'data' );
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
                name = 'JumpGRF';
                setup.data.class = @JumpGRFDataset;
                setup.data.args.Normalization = 'PAD';
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.ResampleRate = 5;
                setup.model.args.trainer.BatchSize = 75;

            case 2
                % Fukuchi ground reaction force in one dimension
                name = 'Fukuchi-GRF-1D';
                setup.data.class = @FukuchiDataset;
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.HasGRF = true;
                setup.data.args.HasVGRFOnly = true;
                setup.data.args.YReference = 'AgeGroup';
                setup.data.args.Category = 'Ground';
                setup.model.args.trainer.BatchSize = 1000;

            case 3
                % Fukuchi hip, knee and ankle joint angles
                name = 'Fukuchi-JointAngles-1D';
                setup.data.class = @FukuchiDataset;
                setup.data.args.HasNormalizedInput = true;
                setup.data.args.YReference = 'AgeGroup';
                setup.data.args.Category = 'JointAngles';
                setup.data.args.HasHipAngles = false;
                setup.data.args.HasKneeAngles = true;
                setup.data.args.HasAnkleAngles = false;
                setup.data.args.SagittalPlaneOnly = true;
                setup.model.args.trainer.BatchSize = 75;
           
            otherwise
                error('Unrecognised dataset ID.');

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


