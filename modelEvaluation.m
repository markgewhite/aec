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
                self = initPCAModel( self, setup );

            else
                % this is an autoencoder model of some kind
                self = initAEModel( self, setup );

            end

            % train the model
            self.Model = train( self.Model, self.TrainingDataset );
            
            % evaluate the trained model
            self.TrainingEvaluation = ...
                modelEvaluation.evaluate( self.Model, self.TrainingDataset );

            self.TestingEvaluation = ...
                modelEvaluation.evaluate( self.Model, self.TestingDataset );

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
            eval.ReconLoss = thisModel.getReconLoss( ...
                                       thisModel, X, eval.XHat );

            % compute the auxiliary loss
            eval.AuxModelYHat = predict( thisModel.auxModel, eval.Z );
            eval.AuxModelLoss = loss( thisModel.auxModel, eval.Z, Y );


        end

    end

end