% Run the analysis of reproducibility using the exemplar data sets

newInvestigation = false;
dataType = 'Real';

% set the results destination
path = fileparts( which('code/reproducibilityAnalysis.m') );
path = [path '/../results/reproducibility/'];
attempt = '004';

% -- data setup --
switch dataType
    case 'Synthetic'
        dataName = 'Exemplar';
        setup.data.class = @ExemplarDataset;   
        setup.data.args.HasNormalizedInput = true;
        setup.data.args.OverSmoothing = 1E8;
        
        % one class, one element
        setup.data.args.ClassSizes = 400;
        setup.data.args.ClassElements = 2;
        setup.data.args.ClassMeans = [ -1.0 1.0 ];
        setup.data.args.ClassSDs = [ 0.5 0.5 ];
        setup.data.args.ClassPeaks = [ 2.0 1.0 ];
        
        % double Gaussian with peak inverse covariance
        setup.data.args.PeakCovariance{1} = [1 -0.5; -0.5 1];
        setup.data.args.MeanCovariance{1} = [1 -0.5; -0.5 1];
        setup.data.args.SDCovariance{1} = [1 0.5; 0.5 1];

    case 'Real'
        dataName = 'JumpGRF';
        setup.data.class = @JumpGRFDataset;   
        setup.data.args.Normalization = 'PAD';
        setup.data.args.HasNormalizedInput = true;
        setup.data.args.NormalizedPts = 51;
        setup.data.args.ResampleRate = 10;
        setup.data.args.OverSmoothing = 1E3;
        setup.data.args.HasMatchingOutput = false;

end

% -- loss functions --
setup.lossFcns.recon.class = @ReconstructionLoss;
setup.lossFcns.recon.name = 'Reconstruction';

setup.lossFcns.adv.class = @AdversarialLoss;
setup.lossFcns.adv.name = 'Discriminator';

% -- model setup --
kfolds = 10;
zdim = 4;
setup.model.args.ZDim = zdim;
setup.model.args.InitZDimActive = 0;
setup.model.args.KFolds = kfolds;
setup.model.args.InitZDimActive = 0;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.RandomSeed = 1234;

% -- trainer setup --
setup.model.args.trainer.NumEpochs = 400;
setup.model.args.trainer.NumEpochsPreTrn = 10;
setup.model.args.trainer.UpdateFreq = 100;
setup.model.args.trainer.BatchSize = 50;
setup.model.args.trainer.Holdout = 0;

% -- grid search --
parameters = [ "model.class" ];
values = { {@FCModel} };

nTests = 7*newInvestigation;

% -- load data, if required --
filename = [dataName '-' num2str(attempt) '-Investigations'];

if newInvestigation
    theInvestigation = cell( nTests,1 ); %#ok<UNRCH> 
else
    if ~exist( 'theInvestigation', 'var' )
        loadData = true;
    elseif isempty( theInvestigation )
        loadData = true;
    else
        loadData = false;
    end
    if loadData
        load( fullfile(path, filename), 'theInvestigation' );
    end
end

for i = 1:nTests

    switch i

        case 1
            % Full control - identical outputs
            name = [ dataName '-' attempt '-FullControl' ];
            setup.model.args.RandomSeedResets = true;
            setup.model.args.IdenticalNetInit = true;
            setup.model.args.trainer.HasMiniBatchShuffle = false;
            setup.model.args.trainer.HasShuffleRandomStream = false;
            setup.model.args.IdenticalPartitions = true;
       
            theInvestigation{i} = Investigation( name, path, parameters, values, setup );

        case 2
            % Allow dropout to vary alone
            name = [ dataName '-' attempt '-DropoutVaries' ];
            setup.model.args.RandomSeedResets = false;
            setup.model.args.IdenticalNetInit = true;
            setup.model.args.trainer.HasMiniBatchShuffle = false;
            setup.model.args.trainer.HasShuffleRandomStream = false;
            setup.model.args.IdenticalPartitions = true;

            theInvestigation{i} = Investigation( name, path, parameters, values, setup );

        case 3
            % Allow mini-batching to vary alone
            name = [ dataName '-' attempt '-MiniBatchVaries' ];
            setup.model.args.RandomSeedResets = true;
            setup.model.args.IdenticalNetInit = true;
            setup.model.args.trainer.HasMiniBatchShuffle = true;
            setup.model.args.trainer.HasShuffleRandomStream = true;
            setup.model.args.IdenticalPartitions = true;
       
            theInvestigation{i} = Investigation( name, path, parameters, values, setup );

        case 4
            % Allow dropout and mini-batching to vary together
            name = [ dataName '-' attempt '-MiniBatch&DropoutVary' ];
            setup.model.args.RandomSeedResets = false;
            setup.model.args.IdenticalNetInit = true;
            setup.model.args.trainer.HasMiniBatchShuffle = true;
            setup.model.args.trainer.HasShuffleRandomStream = false;
            setup.model.args.IdenticalPartitions = true;
       
            theInvestigation{i} = Investigation( name, path, parameters, values, setup );
        
        case 5
            % Allow network initialization to vary alone
            name = [ dataName '-' attempt '-NetInitVaries' ];
            setup.model.args.RandomSeedResets = true;
            setup.model.args.IdenticalNetInit = false;
            setup.model.args.trainer.HasMiniBatchShuffle = false;
            setup.model.args.trainer.HasShuffleRandomStream = false;
            setup.model.args.IdenticalPartitions = true;
       
            theInvestigation{i} = Investigation( name, path, parameters, values, setup );

        case 6
            % Allow data partitions to vary alone
            name = [ dataName '-' attempt '-DataVaries' ];
            setup.model.args.RandomSeedResets = true;
            setup.model.args.IdenticalNetInit = true;
            setup.model.args.trainer.HasMiniBatchShuffle = false;
            setup.model.args.trainer.HasShuffleRandomStream = false;
            setup.model.args.IdenticalPartitions = false;
       
            theInvestigation{i} = Investigation( name, path, parameters, values, setup );

        case 7
            % Allow data partitions to vary alone
            name = [ dataName '-' attempt '-AllVariesExceptData' ];
            setup.model.args.RandomSeedResets = false;
            setup.model.args.IdenticalNetInit = false;
            setup.model.args.trainer.HasMiniBatchShuffle = true;
            setup.model.args.trainer.HasShuffleRandomStream = false;
            setup.model.args.IdenticalPartitions = true;
       
            theInvestigation{i} = Investigation( name, path, parameters, values, setup );

    end

end

if newInvestigation
    save( fullfile( path, filename ), 'theInvestigation' );
end

% conduct an analysis of the outputs
% ----------------------------------
nTests = length( theInvestigation );

% compare summary metrics across folds for each investigation
nComparisons = kfolds*(kfolds-1)/2;
reconLossTrnRMSE = zeros( nTests, 1 );
ZCorrTrnRMSE = zeros( nTests, 1 );
XCCorrTrnRMSE = zeros( nTests, 1 );
for i = 1:nTests
    thisModel = theInvestigation{i}.Evaluations{1}.Model;

    for k1 = 1:kfolds
        thisK1Model = thisModel.SubModels{k1};
        
        for k2 = k1+1:kfolds
            thisK2Model = thisModel.SubModels{k2};

            reconLossTrnRMSE(i) = reconLossTrnRMSE(i) + ...
                    (thisK2Model.Loss.Training.ReconLoss ...
                        - thisK1Model.Loss.Training.ReconLoss).^2;
            
            ZCorrTrnRMSE(i) = ZCorrTrnRMSE(i) + ...
                    (thisK2Model.Correlations.Training.ZCorrelation ...
                        - thisK1Model.Correlations.Training.ZCorrelation).^2;
            
            XCCorrTrnRMSE(i) = XCCorrTrnRMSE(i) + ...
                    (thisK2Model.Correlations.Training.XCCorrelation ...
                        - thisK1Model.Correlations.Training.XCCorrelation).^2;
        
        end
    end
end
reconLossTrnRMSE = sqrt(reconLossTrnRMSE/nComparisons);
ZCorrTrnRMSE = sqrt(ZCorrTrnRMSE/nComparisons);
XCCorrTrnRMSE = sqrt(XCCorrTrnRMSE/nComparisons);


% compare components across folds for each investigation
% the problem is not tractable with brute force
% a genetic algorithm is required to find the best component order
% arrangement across the sub-models

permOrderIdx = perms( 1:zdim );
lb = [ length(permOrderIdx) ones( 1, kfolds-1 ) ];
ub = length(permOrderIdx)*ones( 1, kfolds );
options = optimoptions( 'ga', ...
                        'PopulationSize', 400, ...
                        'EliteCount', 80, ...
                        'MaxGenerations', 300, ...
                        'MaxStallGenerations', 150, ...
                        'FunctionTolerance', 1E-6, ...
                        'UseVectorized', true, ...
                        'PlotFcn', {'gaplotbestf','gaplotdistance', ...
                                    'gaplotbestindiv' } );

componentMSE = zeros( nTests, 1 );
componentPerms = zeros( nTests, kfolds );
componentOrder = zeros( kfolds, zdim, nTests );

compSize = size( theInvestigation{i}.Evaluations{1}.Model.LatentComponents );
latentComp = zeros( compSize(1), zdim, kfolds );

for i = 1:nTests

    thisModel = theInvestigation{i}.Evaluations{1}.Model;
    
    % pre-compile latent components across the sub-models
    % summarising all component lines into the mean absolute curve
    for j = 1:kfolds
        comp = reshape( thisModel.SubModels{j}.LatentComponents, ...
                        compSize(1), thisModel.NumCompLines, [] );
        comp = mean( abs(comp), 2 );
        latentComp(:,:,j) = reshape( comp, compSize(1), [] );
    end
    
    % setup the objective function
    objFcn = @(p) arrangementError( p, latentComp, zdim );
    
    % run the genetic algorithm optimization
    [ componentPerms(i,:), componentMSE(i,:) ] = ...
        ga( objFcn, kfolds, [], [], [], [], lb, ub, [], 1:kfolds, options );

    % generate the order from list of permutations
    for j = 1:kfolds
        componentOrder( j, :, i ) = permOrderIdx( componentPerms(i,j), : );
    end

end

componentRMSE = sqrt( componentMSE(:,1) );
