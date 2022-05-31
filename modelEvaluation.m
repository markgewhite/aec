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


        function self = run( self, setup, verbose )
            arguments
                self            modelEvaluation
                setup           struct
                verbose         logical
            end

            % store the specified setup
            self.Setup = setup;

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
                    tSpan = self.TrainingDataset.tSpan.input, ...
                    PaddingLength = self.TrainingDataset.padding.length );

            if isfield( setup.lossFcns, 'cls' )
                % set the number of classes
                setup.lossFcns.cls.args.CDim = self.TestingDataset.CDim;
            end

            if isequal( setup.model.class, @pcaModel )
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
                disp('Trainer setup:')
                disp( setup.trainer.args );
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
            self.TrainingEvaluation = ...
                modelEvaluation.evaluate( self.Model, self.TrainingDataset );

            self.TestingEvaluation = ...
                modelEvaluation.evaluate( self.Model, self.TestingDataset );

            if verbose
                disp('Training evaluation:');
                disp(['         Reconstruction Loss = ' ...
                    num2str( self.TrainingEvaluation.ReconLoss, '%.3f' )]);
                disp(['Smoothed Reconstruction Loss = ' ...
                    num2str( self.TrainingEvaluation.ReconLossSmoothed, '%.3f' )]);
                disp([' Regular Reconstruction Loss = ' ...
                    num2str( self.TrainingEvaluation.ReconLossRegular, '%.3f' )]);
                disp(['        Auxiliary Model Loss = ' ...
                    num2str( self.TrainingEvaluation.AuxModelLoss, '%.3f' )]);

                disp('Testing evaluation:');
                disp(['         Reconstruction Loss = ' ...
                    num2str( self.TestingEvaluation.ReconLoss, '%.3f' )]);
                disp(['Smoothed Reconstruction Loss = ' ...
                    num2str( self.TrainingEvaluation.ReconLossSmoothed, '%.3f' )]);
                disp([' Regular Reconstruction Loss = ' ...
                    num2str( self.TestingEvaluation.ReconLossRegular, '%.3f' )]);
                disp(['        Auxiliary Model Loss = ' ...
                    num2str( self.TestingEvaluation.AuxModelLoss, '%.3f' )]);
            end


            if verbose
                % plot latent space
                self.Model.plotZDist( self.TestingEvaluation.Z );
                self.Model.plotZClusters( self.TestingEvaluation.Z, ...
                                          Y = self.TestingDataset.Y );
    
                % plot the components
                self.Model.plotLatentComp( ...
                              self.TestingEvaluation.XC, ...
                              self.TestingDataset.tSpan, ...
                              self.TestingDataset.fda.fdParamsTarget, ...
                              type = 'Smoothed', ...
                              shading = true, ...
                              plotTitle = self.TestingDataset.info.datasetName, ...
                              xAxisLabel = self.TestingDataset.info.timeLabel, ...
                              yAxisLabel = self.TestingDataset.info.channelLabels, ...
                              yAxisLimits = self.TestingDataset.info.channelLimits );
            end

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

            self.Model = pcaModel( ...
                            self.TrainingDataset.XChannels, ...
                            setup.model.args.ZDim, ...
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
                            self.TrainingDataset.XTargetDim, ...
                            self.TrainingDataset.XChannels, ...
                            setup.model.args.ZDim, ...
                            self.TrainingDataset.CDim, ...
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
            eval.Z = thisModel.encode( thisDataset );

            % reconstruct the curves
            eval.XHat = squeeze( thisModel.reconstruct( eval.Z ) );

            % smooth the reconstructed curves
            XHatFd = smooth_basis( thisDataset.tSpan.target, ...
                                   eval.XHat, ...
                                   thisDataset.fda.fdParamsTarget );
            eval.XHatSmoothed = squeeze( ...
                        eval_fd( thisDataset.tSpan.target, XHatFd ) );
            
            eval.XHatRegular = squeeze( ...
                        eval_fd( thisDataset.tSpan.regular, XHatFd ) );

            % compute reconstruction loss
            eval.ReconLoss = thisModel.getReconLoss( ...
                                        thisDataset.XTarget, eval.XHat );
            eval.ReconLossSmoothed = ...
                thisModel.getReconLoss( eval.XHatSmoothed, eval.XHat );

            % compute reconstruction loss for the regularised curves
            eval.XRegular = squeeze( thisDataset.XInputRegular );
            eval.ReconLossRegular = ...
                thisModel.getReconLoss( eval.XHatRegular, eval.XRegular );

            % compute the mean squared error as a function of time
            eval.ReconTimeMSE = ...
                thisModel.getReconTemporalLoss( eval.XHatRegular, eval.XRegular );

            figure(4);
            hold on;
            for i = 1:thisDataset.XChannels
                plot( thisDataset.tSpan.regular, eval.ReconTimeMSE(:,i) );
            end


            % compute the auxiliary loss
            ZLong = reshape( eval.Z, size( eval.Z, 1 ), [] );
            eval.AuxModelYHat = predict( thisModel.auxModel, ZLong );
            eval.AuxModelLoss = loss( thisModel.auxModel, ...
                                            ZLong, thisDataset.Y );

            % generate the latent components
            eval.XC = thisModel.latentComponents( ...
                            eval.Z, ...
                            sampling = 'Fixed', ...
                            centre = false, ...
                            convert = true );

            % compute the components' explained variance
            [eval.VarProp, eval.CompVar] = ...
                            thisModel.getExplainedVariance( thisDataset );


        end

    end

end