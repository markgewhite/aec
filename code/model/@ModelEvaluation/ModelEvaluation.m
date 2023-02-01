classdef ModelEvaluation
    % Class defining a model evaluation

    properties
        Name                % name of the evaluation
        Path                % folder for storing results
        BespokeSetup        % structure recording the bespoke setup
        TrainingDataset     % training dataset object
        TestingDataset      % testing dataset object
        KFolds              % number of partitions
        KFoldRepeats        % number of k-fold repetitions
        Partitions          % training dataset k-fold partitions
        CVType              % type of cross-validation
        HasIdenticalPartitions % flag for special case of identical partitions
        NumModels           % number of models
        Models              % trained model objects
        LossFcns            % array of loss function objects
        CVComponents        % cross-validated latent components
        CVAuxMetrics        % structure of auxiliary model/network metrics
        CVLoss              % structure of cross-validated losses
        CVCorrelations      % structure of cross-validated losses
        CVTiming            % structure of cross-validated execution times
        ComponentOrder      % optimal arrangement of model components
        ComponentDiffRMSE   % overall difference between sub-models
        RandomSeed          % for reproducibility
        RandomSeedResets    % whether to reset the seed for each model
    end


    methods


        function self = ModelEvaluation( name, path, setup, args )
            % Construct and run a model evaluation object
            arguments
                name                    string
                path                    string {mustBeFolder}
                setup                   struct
                args.CVType             string ...
                        {mustBeMember( args.CVType, ...
                        {'Holdout', 'KFold'} )} = 'Holdout'
                args.KFolds             double ...
                        {mustBeInteger, mustBePositive} = 1
                args.KFoldRepeats       double ...
                        {mustBeInteger, mustBePositive} = 1
                args.HasIdenticalPartitions logical = false
                args.RandomSeed         double ...
                        {mustBeInteger, mustBePositive} = 1234
                args.RandomSeedResets   logical = false;
            end

            % store the name for this evaluation and its bespoke setup
            self.Name = name;
            self.Path = path;
            self.BespokeSetup = setup;

            % store other arguments
            self.CVType = args.CVType;
            self.KFolds = args.KFolds;
            self.KFoldRepeats = args.KFoldRepeats;
            self.HasIdenticalPartitions = args.HasIdenticalPartitions;
            self.RandomSeed = args.RandomSeed;
            self.RandomSeedResets = args.RandomSeedResets;

            if ~isempty( self.RandomSeed )
                % set random seed for reproducibility
                rng( self.RandomSeed );
            end

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

            % prepare the data
            self = initDatasets( self, setup );
            
            % train the model
            self = self.trainModels( setup.model );

            % evaluate the trained model
            self = self.evaluateModels( 'Training' );
            self = self.evaluateModels( 'Testing' );           

            disp('Training evaluation:');
            reportResult( self.CVLoss.Training.Mean, ...
                          self.CVCorrelations.Training.Mean );
            disp('Testing evaluation:');
            reportResult( self.CVLoss.Testing.Mean, ...
                          self.CVCorrelations.Testing.Mean );

        end


        % methods

        self = arrangeComponents( self )

        XC = calcCVComponents( self )

        self = conserveMemory( self, level )

        self = initPCAModel( self, setup )

        self = initAEModel( self, setup )

        save( self )

        saveReport( self )


    end

end