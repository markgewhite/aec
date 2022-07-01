% Run the analysis of reproducibility using the exemplar data sets

newInvestigation = false;

% set the results destination
path = fileparts( which('code/reproducibilityAnalysis.m') );
path = [path '/../results/reproducibility/'];
attempt = '003';

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
        filename = ['Exemplar-' num2str(attempt) '-Investigations'];
        load( fullfile(path, filename), 'theInvestigation' );
    end
end

% -- data setup --
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
setup.model.args.randomSeed = 1234;

% -- trainer setup --
setup.model.args.trainer.numEpochs = 400;
setup.model.args.trainer.numEpochsPreTrn = 10;
setup.model.args.trainer.updateFreq = 100;
setup.model.args.trainer.batchSize = 50;
setup.model.args.trainer.holdout = 0;

% -- grid search --
parameters = [ "model.class" ];
values = { {@FCModel} };

nTests = 6*newInvestigation;

for i = 1:nTests

    switch i

        case 1
            % Full control - identical outputs
            name = [ 'Exemplar-' attempt '-FullControl' ];
            setup.model.args.randomSeedResets = true;
            setup.model.args.identicalNetInit = true;
            setup.model.args.trainer.miniBatchShuffle = false;
            setup.model.args.trainer.randomStatePreservedOnShuffle = false;
            setup.model.args.IdenticalPartitions = true;
       
            theInvestigation{i} = ...
                Investigation( name, path, parameters, values, setup );
            theInvestigation{i} = theInvestigation{i}.clearPredictions;

        case 2
            % Allow dropout to vary alone
            name = [ 'Exemplar-' attempt '-DropoutVaries' ];
            setup.model.args.randomSeedResets = false;
            setup.model.args.identicalNetInit = true;
            setup.model.args.trainer.miniBatchShuffle = false;
            setup.model.args.trainer.randomStatePreservedOnShuffle = false;
            setup.model.args.IdenticalPartitions = true;

            theInvestigation{i} = ...
                Investigation( name, path, parameters, values, setup );
            theInvestigation{i} = theInvestigation{i}.clearPredictions;

        case 3
            % Allow mini-batching to vary alone
            name = [ 'Exemplar-' attempt '-MiniBatchVaries' ];
            setup.model.args.randomSeedResets = true;
            setup.model.args.identicalNetInit = true;
            setup.model.args.trainer.miniBatchShuffle = true;
            setup.model.args.trainer.randomStatePreservedOnShuffle = true;
            setup.model.args.IdenticalPartitions = true;
       
            theInvestigation{i} = ...
                Investigation( name, path, parameters, values, setup );
            theInvestigation{i} = theInvestigation{i}.clearPredictions;

        case 4
            % Allow dropout and mini-batching to vary together
            name = [ 'Exemplar-' attempt '-MiniBatch&DropoutVary' ];
            setup.model.args.randomSeedResets = false;
            setup.model.args.identicalNetInit = true;
            setup.model.args.trainer.miniBatchShuffle = true;
            setup.model.args.trainer.randomStatePreservedOnShuffle = false;
            setup.model.args.IdenticalPartitions = true;
       
            theInvestigation{i} = ...
                Investigation( name, path, parameters, values, setup );
            theInvestigation{i} = theInvestigation{i}.clearPredictions;
        
        case 5
            % Allow network initialization to vary alone
            name = [ 'Exemplar-' attempt '-NetInitVaries' ];
            setup.model.args.randomSeedResets = true;
            setup.model.args.identicalNetInit = false;
            setup.model.args.trainer.miniBatchShuffle = false;
            setup.model.args.trainer.randomStatePreservedOnShuffle = false;
            setup.model.args.IdenticalPartitions = true;
       
            theInvestigation{i} = ...
                Investigation( name, path, parameters, values, setup );
            theInvestigation{i} = theInvestigation{i}.clearPredictions;

        case 6
            % Allow data partitions to vary alone
            name = [ 'Exemplar-' attempt '-DataVaries' ];
            setup.model.args.randomSeedResets = true;
            setup.model.args.identicalNetInit = true;
            setup.model.args.trainer.miniBatchShuffle = false;
            setup.model.args.trainer.randomStatePreservedOnShuffle = false;
            setup.model.args.IdenticalPartitions = false;
       
            theInvestigation{i} = ...
                Investigation( name, path, parameters, values, setup );
            theInvestigation{i} = theInvestigation{i}.clearPredictions;

    end

end


% conduct an analysis of the outputs
% ----------------------------------
nTests = 6;

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
p = perms(1:zdim);
nCompComparisons = length(p);
nLinesPerComponent = size( theInvestigation{1}.Evaluations{1}.Model.LatentComponents, 2 )/zdim;
componentRMSE = zeros( nTests, nComparisons, nCompComparisons );
componentRMSEBest = zeros( nTests, nComparisons );

for i = 1:nTests
    thisModel = theInvestigation{i}.Evaluations{1}.Model;
    
    j = 0;
    for k1 = 1:kfolds
        thisK1Model = thisModel.SubModels{k1};
        
        for k2 = k1+1:kfolds
            thisK2Model = thisModel.SubModels{k2};

            j = j+1;
            for q = 1:length(p)
                    
                for d = 1:zdim
                    c1A = (d-1)*nLinesPerComponent + 1;
                    c1B = c1A + nLinesPerComponent - 1;

                    c2A = (p(q,d)-1)*nLinesPerComponent + 1;
                    c2B = c2A + nLinesPerComponent - 1;

                    compC1 = thisK1Model.LatentComponents(:,c1A:c1B);
                    compC2 = thisK2Model.LatentComponents(:,c2A:c2B);
                                            
                    componentRMSE(i,j,q) = componentRMSE(i,j,q) + ...
                        mean( (compC1 - compC2).^2, 'all' );
                end
                componentRMSE(i,j,q) = componentRMSE(i,j,q)/zdim;

            end
            componentRMSEBest(i,j) = min( componentRMSE(i,j,:) );

        end
    end
end
componentRMSEBest = sqrt( mean(componentRMSEBest, 2) );
