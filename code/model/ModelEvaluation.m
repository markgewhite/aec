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
                    PaddingLength = self.TrainingDataset.Padding.Length );

            if isequal( setup.model.class, @FullPCAModel )
                % this is a PCA
                if verbose
                    disp('********* PCA Model Evaluation *********');
                end
                self = initPCAModel( self, setup );

            else
                % this is an autoencoder model of some kind
                if verbose
                    disp('***** Autoencoder Model Evaluation *****');
                end
                self = initAEModel( self, setup );

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


        function plotModel( self )
            % Display the full model plots
            arguments
                self        ModelEvaluation
            end

            if ~empty( self.Model.Figs ) && ~empty( self.Model.Axes )

                % plot latent space
                plotZDist( self.Model, self.TestingPredictions.Z );
                plotZClusters( self.Model, self.TestingPredictions.Z, ...
                                           Y = self.TestingDataset.Y );
                % plot the components
                plotLatentComp( self.Model, type = 'Smoothed', shading = true );
            
            else
                % graphics objects must have been cleared
                eid = 'Evaluation:NoGrahicsObjects';
                msg = 'There are no graphics objects specified in the model.';
                throwAsCaller( MException(eid,msg) );

            end

        end

        
        function save( self, path, name )
            % Save the evaluation to a specified path
            arguments
                self        ModelEvaluation
                path        string {mustBeFolder}
                name        string
            end

            % define a small structure for saving
            output.BespokeSetup = self.BespokeSetup;
            output.TrainingEvaluation = self.TrainingEvaluation;
            output.TestingEvaluation = self.TestingEvaluation;
            output.TrainingCorrelations = self.TrainingCorrelations;
            output.TestingCorrelations = self.TestingCorrelations;
            
            name = strcat( name, "-OverallEvaluation" );
            save( fullfile( path, name ), 'output' );

        end         

    end


    methods (Access = private)

        function self = initPCAModel( self, setup )
            % Initialize a PCA model
            arguments
                self    ModelEvaluation
                setup   struct
            end

            % limit the arguments to relevant fields
            pcaFields = {'KFolds', 'IdenticalPartitions', ...
                         'ZDim', 'AuxModelType', 'name', 'path'};
            for i = 1:length(pcaFields)
                if isfield( setup.model.args, pcaFields{i} )
                    args.(pcaFields{i}) = setup.model.args.(pcaFields{i});
                end
            end

            try
                argsCell = namedargs2cell( args );
            catch
                argsCell = {};
            end

            self.Model = FullPCAModel( self.TrainingDataset, argsCell{:} ); 

        end


        function self = initAEModel( self, setup )
            % Initialize an autoencoder model
            arguments
                self    ModelEvaluation
                setup   struct
            end

            % initialize the loss functions
            lossFcnNames = fields( setup.lossFcns );
            nLossFcns = length( lossFcnNames );
            self.LossFcns = cell( nLossFcns, 1 );

            for i = 1:nLossFcns
                
                thelossFcn = setup.lossFcns.(lossFcnNames{i});
                
                try
                    argsCell = namedargs2cell( thelossFcn.args );
                catch
                    argsCell = {};
                end

                self.LossFcns{i} = ...
                        thelossFcn.class( thelossFcn.name, argsCell{:} );
            end

            % initialize the model
            try
                argsCell = namedargs2cell( setup.model.args );
            catch
                argsCell = {};
            end
            self.Model = setup.model.class( self.TrainingDataset, ...
                                            self.LossFcns{:}, ...
                                            argsCell{:} );

        end


    end

end


function reportResult( eval, cor )

    disp(['         Reconstruction Loss = ' ...
        num2str( eval.ReconLoss, '%.3f' )]);
    disp(['Smoothed Reconstruction Loss = ' ...
        num2str( eval.ReconLossSmoothed, '%.3f' )]);
    disp([' Regular Reconstruction Loss = ' ...
        num2str( eval.ReconLossRegular, '%.3f' )]);
    disp(['  Auxiliary Model Error Rate = ' ...
        num2str( eval.AuxModelErrorRate, '%.3f' )]);
    disp(['    Auxiliary Model F1 Score = ' ...
        num2str( eval.AuxModelF1Score, '%.3f' )]);

    if isfield( eval, 'AuxNetworkErrorRate' )
        disp(['Auxiliary Network Error Rate = ' ...
            num2str( eval.AuxNetworkErrorRate, '%.3f' )]);
        disp(['  Auxiliary Network F1 Score = ' ...
            num2str( eval.AuxNetworkF1Score, '%.3f' )]);
    end

    if isfield( eval, 'ComparatorErrorRate' )
        disp(['       Comparator Error Rate = ' ...
            num2str( eval.ComparatorErrorRate, '%.3f' )]);
        disp(['         Comparator F1 Score = ' ...
            num2str( eval.ComparatorF1Score, '%.3f' )]);
    end

    disp(['               Z Correlation = ' ...
        num2str( cor.ZCorrelation, '%.3f' )]);
    disp(['              XC Correlation = ' ...
        num2str( cor.XCCorrelation, '%.3f' )]);

end