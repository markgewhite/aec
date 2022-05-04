classdef modelEvaluation < handle
    % Class defining a model evaluation

    properties
        Setup               % structure recording the setup
        TrainingDataset     % training dataset object
        TestingDataset      % testing dataset object
        Model               % trained model object
        LossFcns            % array of loss function objects
        TrainingEvaluation  % training data's evaluation
        TestingEvaluation   % testing data's evaluation
    end


    methods

        function self = modelEvaluation
            % Construct a model evaluation object (placeholder)

        end


        function self = run( self, setup )
            arguments
                self            modelEvaluation
                setup           struct
            end

            % store the specified setup
            self.Setup = setup;

            % prepare data
            self.TrainingDataset = setup.data.class( 'Training', ...
                                     normalization = 'PAD', ...
                                     normalizeInput = true ); %#ok<*MCNPN> 

            self.TestingDataset = setup.data.class( 'Testing', ...
                                     normalization = 'PAD', ...
                                     normalizeInput = true );

            if isequal( setup.model.class, @pcaModel )
                % this is a PCA
                disp('********* PCA Model Evaluation *********');
                self = initPCAModel( self, setup );

            else
                % this is an autoencoder model of some kind
                disp('***** Autoencoder Model Evaluation *****');
                self = initAEModel( self, setup );

            end

            % display setup
            disp('Data setup:')
            disp( setup.data );
            disp('Model setup:')
            disp( setup.model.class );
            disp( setup.model.args );
            disp('Trainer setup:')
            disp( setup.trainer.args );

            % train the model
            disp('Training the model ...');
            self.Model = train( self.Model, self.TrainingDataset );
            disp('Training complete');
            
            % evaluate the trained model
            disp('Training evaluation:');
            self.TrainingEvaluation = ...
                modelEvaluation.evaluate( self.Model, self.TrainingDataset );

            disp('Testing evaluation:');
            self.TestingEvaluation = ...
                modelEvaluation.evaluate( self.Model, self.TestingDataset );

            % plot latent space
            self.Model.plotZDist( self.TestingEvaluation.Z );
            self.Model.plotZClusters( self.TestingEvaluation.Z, ...
                                      Y = self.TestingDataset.Y );

            % plot the components
            self.Model.plotLatentComp( ...
                          self.TestingEvaluation.XC, ...
                          self.TestingDataset.fda, ...
                          type = 'Smoothed', ...
                          shading = true, ...
                          plotTitle = self.TestingDataset.info.datasetName, ...
                          xAxisLabel = self.TestingDataset.info.timeLabel, ...
                          yAxisLabel = self.TestingDataset.info.channelLabels, ...
                          yAxisLimits = self.TestingDataset.info.channelLimits );
        end


        function save( self, path, name )
            % Save the evaluation to a specified path
            arguments
                self        modelEvaluation
                path        string {mustBeFolder}
                name        string
            end

            % save the Z distribution plot
            fullname = strcat( name, '.pdf' );
            fullpath = strcat( path, '/zdist/' );
            if ~isfolder( fullpath )
                mkdir( fullpath)
            end
            exportgraphics( self.Model.Axes.ZDistribution, ...
                            fullfile( fullpath, fullname ), ...
                            ContentType= 'vector', ...
                            Resolution = 300 );

            % save the Z clustering plot
            fullpath = strcat( path, '/zclust/' );
            if ~isfolder( fullpath )
                mkdir( fullpath)
            end
            exportgraphics( self.Model.Axes.ZClustering, ...
                            fullfile( fullpath, fullname ), ...
                            ContentType= 'vector', ...
                            Resolution = 300 );


            % save the loss plots
            if isa( self.Model, 'autoencoderModel' )
                fullpath = strcat( path, '/loss/' );
                if ~isfolder( fullpath )
                    mkdir( fullpath)
                end
                exportgraphics( self.Model.trainer.lossFig, ...
                            fullfile( fullpath, fullname ), ...
                            ContentType= 'vector', ...
                            Resolution = 300 );
            end

            % save the components (as a whole and individually)
            fullpath = strcat( path, '/comp/' );
            if ~isfolder( fullpath )
                mkdir( fullpath)
            end

            % all components
            exportgraphics( self.Model.Figs.Components, ...
                            fullfile( fullpath, fullname ), ...
                            ContentType= 'vector', ...
                            Resolution = 300 );

            % individual components
            for i = 1:length(self.Model.Axes.Comp)

                fullname = strcat( name, num2str(i), '.pdf' );
                exportgraphics( self.Model.Axes.Comp(i), ...
                            fullfile( fullpath, fullname ), ...
                            ContentType= 'vector', ...
                            Resolution = 300 );


            end

        end

    end


    methods (Access = private)

        function self = initPCAModel( self, setup )
            % Initialize a PCA model
            arguments
                self    modelEvaluation
                setup   struct
            end

            % limit the arguments to two possible fields
            if isfield( setup.model.args, 'ZDim' )
                args.ZDim = setup.model.args.ZDim;
            end
            if isfield( setup.model.args, 'auxModel' )
                args.auxModel = setup.model.args.auxModel;
            end

            try
                argsCell = namedargs2cell( args );
            catch
                argsCell = {};
            end

            self.Model = pcaModel( self.TrainingDataset.fda.fdParams, ...
                                   argsCell{:} ); 

        end


        function self = initAEModel( self, setup )
            % Initialize an autoencoder model
            arguments
                self    modelEvaluation
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
            self.Model = setup.model.class( ...
                            self.TrainingDataset.XInputDim, ...
                            self.TrainingDataset.XChannels, ...
                            self.LossFcns{:}, ...
                            argsCell{:} );

            % initialize the trainer
            try
                argsCell = namedargs2cell( setup.trainer.args );
            catch
                argsCell = {};
            end
            self.Model = self.Model.initTrainer( argsCell{:} );

            % initialize the optimizer
            try
                argsCell = namedargs2cell( setup.optimizer.args );
            catch
                argsCell = {};
            end
            self.Model = self.Model.initOptimizer( argsCell{:} );


        end


    end





    methods (Static, Access = private)

        function eval = evaluate( thisModel, thisDataset )
            % Evaluate the model with a specified dataset
            % (training/testing)
            arguments
                thisModel       representationModel
                thisDataset     modelDataset
            end

            % generate latent encoding using the trained model
            eval.Z = thisModel.encode( thisModel, thisDataset );

            % reconstruct the curves
            eval.XHat = thisModel.reconstruct( thisModel, eval.Z );

            % compute reconstruction loss
            [X, Y] = thisDataset.getInput( dlarray=false );
            eval.ReconLoss = thisModel.getReconLoss( X, eval.XHat );
            disp(['Reconstruction Loss = ' ...
                            num2str( eval.ReconLoss, '%.3f' )]);

            % compute the auxiliary loss
            eval.AuxModelYHat = predict( thisModel.auxModel, eval.Z );
            eval.AuxModelLoss = loss( thisModel.auxModel, eval.Z, Y );
            disp(['Auxiliary Model Loss = ' ...
                            num2str( eval.AuxModelLoss, '%.3f' )]);

            % generate the latent components
            eval.XC = thisModel.latentComponents( ...
                            eval.Z, ...
                            sampling = 'Fixed', ...
                            centre = false );


        end

    end

end