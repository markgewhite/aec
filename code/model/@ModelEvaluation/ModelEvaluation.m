classdef ModelEvaluation
    % Class defining a model evaluation

    properties
        Name                % name of the evaluation
        BespokeSetup        % structure recording the bespoke setup
        TrainingDataset     % training dataset object
        TestingDataset      % testing dataset object
        KFolds              % number of partitions
        Partitions          % training dataset k-fold partitions
        CVType              % type of cross-validation
        HasIdenticalPartitions % flag for special case of identical partitions
        Models              % trained model objects
        LossFcns            % array of loss function objects
        CVComponents        % cross-validated latent components
        CVAuxMetrics        % structure of auxiliary model/network metrics
        ComponentOrder      % optimal arrangement of sub-model components
        ComponentDiffRMSE   % overall difference between sub-models
        TrainingEvaluation  % cross-validated training evaluation
        TestingEvaluation   % cross-validated testing evaluation
        TrainingPredictions % ensemble training predictions
        TestingPredictions  % ensemble testing evaluation
        TrainingCorrelations % ensemble training correlations
        TestingCorrelations  % ensemble testing correlations
        RandomSeed          % for reproducibility
        RandomSeedResets    % whether to reset the seed for each model
    end


    methods


        function self = ModelEvaluation( name, setup, args )
            % Construct and run a model evaluation object
            arguments
                name                    string
                setup                   struct
                args.CVType             string ...
                        {mustBeMember( args.CVType, ...
                        {'Holdout', 'KFold'} )} = 'Holdout'
                args.KFolds             double ...
                        {mustBeInteger, mustBePositive} = 1
                args.HasIdenticalPartitions logical = false
                args.RandomSeed         double ...
                        {mustBeInteger, mustBePositive} = 1234
                args.RandomSeedResets   logical = false;
                args.verbose            logical = true
            end

            % store the name for this evaluation and its bespoke setup
            self.Name = name;
            self.BespokeSetup = setup;

            % store other arguments
            self.CVType = args.CVType;
            self.KFolds = args.KFolds;
            self.HasIdenticalPartitions = args.HasIdenticalPartitions;
            self.RandomSeed = args.RandomSeed;
            self.RandomSeedResets = args.RandomSeedResets;

            if ~isempty( self.RandomSeed )
                % set random seed for reproducibility
                rng( self.RandomSeed );
            end

            % prepare the data
            self = initDatasets( self, setup );

            if args.verbose 
                if isequal( setup.model.class, @PCAModel )
                    disp('********* PCA Model Evaluation *********');
                    setup.model.args = trimPCAArgs( setup.model.args );
                else
                    disp('***** Autoencoder Model Evaluation *****');
                end
                disp('Data setup:')
                disp( setup.data.class );
                disp( setup.data.args );
                disp('Model setup:')
                disp( setup.model.class );
                disp( setup.model.args );
            end

            % train the model
            if args.verbose
                disp('Training the model ...');
            end
            self.trainModels( self.TrainingDataset, setup.model );
            if args.verbose
                disp('Training complete');
            end

            % evaluate the trained model
            [ self.TrainingEvaluation, ...
                self.TrainingPredictions, ...
                    self.TrainingCorrelations ] ...
                    = self.Models.evaluateSet( self.TrainingDataset );

            [ self.TestingEvaluation, ...
                self.TestingPredictions, ...
                    self.TestingCorrelations ] ...
                    = self.Models.evaluateSet( self.TestingDataset );

            if args.verbose
                disp('Training evaluation:');
                reportResult( self.TrainingEvaluation, ...
                              self.TrainingCorrelations );
                disp('Testing evaluation:');
                reportResult( self.TestingEvaluation, ...
                              self.TestingCorrelations );
            end

        end


        % methods

        self = arrangeComponents( self )

        XC = calcCVComponents( self )

        self = initPCAModel( self, setup )

        self = initAEModel( self, setup )

        plotModel( self )

        save( self, path, name )


    end

end