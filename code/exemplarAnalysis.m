% Run the analysis for the exemplar data sets

clear;

% set the results destination
path = fileparts( which('code/exemplarAnalysis.m') );
path = [path '/../results/exemplars/'];

% -- data setup --
setup.data.class = @ExemplarDataset;   
setup.data.args.HasNormalizedInput = true;
setup.data.args.OverSmoothing = 1E8;

% -- loss functions --
lossFcns1.recon.class = @ReconstructionLoss;
lossFcns1.recon.name = 'Reconstruction';
lossFcns1.adv.class = @AdversarialLoss;
lossFcns1.adv.name = 'Discriminator';

lossFcns2 = lossFcns1;
lossFcns2.zcls.class = @ClassifierLoss;
lossFcns2.zcls.name = 'ZClassifier';
lossFcns2.xcls.class = @ComparatorLoss;
lossFcns2.xcls.name = 'XClassifier';

setup.lossFcns = lossFcns1;

% -- model setup --
setup.model.args.InitZDimActive = 0;
setup.model.args.KFolds = 1;
setup.model.args.AuxModel = 'Logistic';
setup.model.args.randomSeed = 1234;

% -- trainer setup --
setup.model.args.trainer.numEpochs = 400;
setup.model.args.trainer.numEpochsPreTrn = 10;
setup.model.args.trainer.updateFreq = 50;
setup.model.args.trainer.batchSize = 1000;
setup.model.args.trainer.holdout = 0;

% -- grid search --
parameters = [ "model.class" "model.args.ZDim" ];
values = {{@FCModel, @ConvolutionalModel, @FullPCAModel} 1:4 }; 
N = 500;
sigma = 0.8;

idx = 1:6;

for i = idx

    switch i

        case {1 2}
            % one class, one element
            setup.data.args.ClassSizes = N;
            setup.data.args.ClassElements = 1;
            setup.data.args.ClassMeans = 0.0;
            setup.data.args.ClassSDs = 0.5;
            setup.data.args.ClassPeaks = 2.0;

        case {3 4}
            % one class, two elements
            setup.data.args.ClassSizes = N;
            setup.data.args.ClassElements = 2;
            setup.data.args.ClassMeans = [ -1.0 1.0 ];
            setup.data.args.ClassSDs = [ 0.5 0.5 ];
            setup.data.args.ClassPeaks = [ 2.0 1.0 ];
    
    end

    switch i

        case 1
            % Single Gaussian with peak (height) variance
            name = 'SingleGaussian-PeakVar';
    
            setup.data.args.PeakCovariance{1} = 1;
            setup.data.args.MeanCovariance{1} = 1E-6;
            setup.data.args.SDCovariance{1} = 1E-6;
       
            singleGaussianPVInvestigation = ...
                Investigation( name, path, parameters, values, setup );

        case 2
            % Single Gaussian with mean (position) variance
            name = 'SingleGaussian-MeanVar';

            setup.data.args.PeakCovariance{1} = 1E-6;
            setup.data.args.MeanCovariance{1} = 1;
            setup.data.args.SDCovariance{1} = 1E-6;
    
            singleGaussianMVInvestigation = ...
                Investigation( name, path, parameters, values, setup );

        case 3
            % Double Gaussian with peak inverse covariance
            name = 'DoubleGaussian-PeakVar';
    
            setup.data.args.PeakCovariance{1} = [1 -sigma; -sigma 1];
            setup.data.args.MeanCovariance{1} = 1E-6*eye(2);
            setup.data.args.SDCovariance{1} = 1E-6*eye(2);
       
            doubleGaussianPVInvestigation = ...
                Investigation( name, path, parameters, values, setup );

        case 4
            % Double Gaussian with peak inverse covariance
            name = 'DoubleGaussian-MeanSDVar';
    
            setup.data.args.PeakCovariance{1} = [1 sigma; sigma 1];
            setup.data.args.MeanCovariance{1} = [1 -sigma; -sigma 1];
            setup.data.args.SDCovariance{1} = [1 sigma; sigma 1];
       
            doubleGaussianMVInvestigation = ...
                Investigation( name, path, parameters, values, setup );

        case 5
            % Two classes each with a single Gaussian
            name = 'SingleGaussian-2Classes';

            setup.data.args.ClassSizes = [ N/2 N/2 ];
            setup.data.args.ClassElements = 1;
            setup.data.args.ClassMeans = [ -1; 1 ];
            setup.data.args.ClassSDs = [ 0.5; 0.5 ];
            setup.data.args.ClassPeaks = [ 2.0; 1.0 ];
    
            setup.data.args.PeakCovariance{1} = 1;
            setup.data.args.MeanCovariance{1} = 1E-6;
            setup.data.args.SDCovariance{1} = 1;

            setup.data.args.PeakCovariance{2} = sigma;
            setup.data.args.MeanCovariance{2} = 1E-6;
            setup.data.args.SDCovariance{2} = 1E-6;
       
            setup.lossFcns = lossFcns1;
            singleGaussian2CInvestigation = ...
                Investigation( name, path, parameters, values, setup );

            setup.lossFcns = lossFcns2;
            singleGaussian2CCInvestigation = ...
                Investigation( name, path, parameters, values, setup );

        case 6
            % Two classes each with a double Gaussian
            name = 'DoubleGaussian-2Classes';

            setup.data.args.ClassSizes = [ N/2 N/2 ];
            setup.data.args.ClassElements = 2;
            setup.data.args.ClassMeans = [ -1 0; 0 1 ];
            setup.data.args.ClassSDs = [ 0.5 0.3; 0.2 0.1 ];
            setup.data.args.ClassPeaks = [ 2.0 3.0; 2.0 1.0 ];
    
            setup.data.args.PeakCovariance{1} = 0.1*[1 -sigma; -sigma 1];
            setup.data.args.MeanCovariance{1} = 0.1*[1 -sigma; -sigma 1];
            setup.data.args.SDCovariance{1} = 0.1*[1 -sigma; -sigma 1];

            setup.data.args.PeakCovariance{2} = 0.2*[1 -sigma; -sigma 1];
            setup.data.args.MeanCovariance{2} = 0.2*[1 -sigma; -sigma 1];
            setup.data.args.SDCovariance{2} = 0.2*[1 -sigma; -sigma 1];
       
            setup.lossFcns = lossFcns1;
            doubleGaussian2CInvestigation = ...
                Investigation( name, path, parameters, values, setup );

            setup.lossFcns = lossFcns2;
            doubleGaussian2CCInvestigation = ...
                Investigation( name, path, parameters, values, setup );
            

    end

end