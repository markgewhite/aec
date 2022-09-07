classdef ModelEvaluation
    % Class defining a model evaluation

    properties
        Name                % name of the evaluation
        BespokeSetup        % structure recording the bespoke setup
        TrainingDataset     % training dataset object
        TestingDataset      % testing dataset object
        Model               % trained model object
        LossFcns            % array of loss function objects
        TrainingEvaluation  % cross-validated training evaluation
        TestingEvaluation   % cross-validated training evaluation
        TrainingPredictions % ensemble training predictions
        TestingPredictions  % ensemble testing evaluation
        TrainingCorrelations % ensemble training correlations
        TestingCorrelations  % ensemble testing correlations
    end


    methods


        function self = ModelEvaluation( name, setup, verbose )
            % Construct and run a model evaluation object
            arguments
                name            string
                setup           struct
                verbose         logical
            end

            % store the name for this evaluation
            self.Name = name;

            % store the specified setup
            self.BespokeSetup = setup;

            % prepare data
            try
                argsCell = namedargs2cell( setup.data.args );
            catch
                argsCell = {};
            end
            self.TrainingDataset = setup.data.class( 'Training', ...
                                            argsCell{:} ); %#ok<*MCNPN> 

            self.TestingDataset = setup.data.class( 'Testing', ...
                                                    argsCell{:}, ...
                    tSpan = self.TrainingDataset.TSpan.Input, ...
                    PaddingLength = self.TrainingDataset.Padding.Length, ...
                    Lambda = self.TrainingDataset.FDA.Lambda );

            if isequal( setup.model.class, @FullPCAModel )
                % this is a PCA
                if verbose
                    disp('********* PCA Model Evaluation *********');
                end
                self = self.initPCAModel( setup );

            else
                % this is an autoencoder model of some kind
                if verbose
                    disp('***** Autoencoder Model Evaluation *****');
                end
                self = self.initAEModel( setup );

            end

            % display setup
            if verbose
                disp('Data setup:')
                disp( setup.data.class );
                disp( setup.data.args );
                disp('Model setup:')
                disp( setup.model.class );
                disp( setup.model.args );
            end

            % train the model
            if verbose
                disp('Training the model ...');
            end
            self.Model = train( self.Model, self.TrainingDataset );
            if verbose
                disp('Training complete');
            end

            % evaluate the trained model
            [ self.TrainingEvaluation, ...
                self.TrainingPredictions, ...
                    self.TrainingCorrelations ] ...
                    = self.Model.evaluateSet( self.TrainingDataset );

            [ self.TestingEvaluation, ...
                self.TestingPredictions, ...
                    self.TestingCorrelations ] ...
                    = self.Model.evaluateSet( self.TestingDataset );

            if verbose
                disp('Training evaluation:');
                reportResult( self.TrainingEvaluation, ...
                              self.TrainingCorrelations );
                disp('Testing evaluation:');
                reportResult( self.TestingEvaluation, ...
                              self.TestingCorrelations );
            end

        end


        % methods

        plotModel( self )

        save( self, path, name )

        self = initPCAModel( self, setup )

        self = initAEModel( self, setup )


    end

end