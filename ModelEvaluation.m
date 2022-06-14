classdef ModelEvaluation < handle
    % Class defining a model evaluation

    properties
        Setup               % structure recording the setup
        TrainingDataset     % training dataset object
        TestingDataset      % testing dataset object
        Model               % trained model object
        LossFcns            % array of loss function objects
        TrainingEvaluation  % cross-validated training evaluation
        TestingEvaluation   % cross-validated training evaluation
        TrainingPredictions % ensemble training predictions
        TestingPredictions  % ensemble testing evaluation
    end


    methods

        function self = ModelEvaluation
            % Construct a model evaluation object (placeholder)

        end


        function self = run( self, setup, verbose )
            arguments
                self            ModelEvaluation
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
            [ self.TrainingEvaluation, self.TrainingPredictions ] ...
                    = ensembleEvaluation( ...
                                    self.Model, self.TrainingDataset );

            [ self.TestingEvaluation, self.TestingPredictions ] ...
                    = ensembleEvaluation( ...
                                    self.Model, self.TestingDataset );

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
                plotZDist( self.Model, self.TestingPredictions.Z );
                plotZClusters( self.Model, self.TestingPredictions.Z, ...
                                           Y = self.TestingDataset.Y );
    
                % plot the components
                plotLatentComp( self.Model, type = 'Smoothed', shading = true );
            end

        end


        function save( self, path, name )
            % Save the evaluation to a specified path
            arguments
                self        ModelEvaluation
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
            if isa( self.Model, 'FullAEModel' )
                fullpath = strcat( path, '/loss/' );
                if ~isfolder( fullpath )
                    mkdir( fullpath)
                end
                exportgraphics( self.Model.Trainer.LossFig, ...
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
                self    ModelEvaluation
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


function [ eval, pred ] = ensembleEvaluation( thisModel, thisDataset )
    % Calculate the aggregate evaluation using all sub-models
    arguments
        thisModel       FullRepresentationModel
        thisDataset     ModelDataset
    end

    % generate latent encodings from all sub-models
    [ pred.ZEnsemble, pred.Z, pred.ZStd ]= ...
                        thisModel.encode( thisDataset );

    % reconstruct the curves
    [ pred.XHatEnsemble, pred.XHat, pred.XHatStd ] = ...
                        thisModel.reconstruct( pred.ZEnsemble );

    % smooth the reconstructed curves
    XHatFd = smooth_basis( thisDataset.TSpan.Target, ...
                           pred.XHat, ...
                           thisDataset.FDA.FdParamsTarget );
    pred.XHatSmoothed = squeeze( ...
                eval_fd( thisDataset.TSpan.Target, XHatFd ) );
    
    pred.XHatRegular = squeeze( ...
                eval_fd( thisDataset.TSpan.Regular, XHatFd ) );

    % compute reconstruction loss
    eval.ReconLoss = reconLoss( thisDataset.XTarget, pred.XHat, ...
                                thisModel.Scale );
    eval.ReconLossSmoothed = reconLoss( pred.XHatSmoothed, pred.XHat, ...
                                        thisModel.Scale );

    % compute reconstruction loss for the regularised curves
    pred.XRegular = squeeze( thisDataset.XInputRegular );
    eval.ReconLossRegular = reconLoss( pred.XHatRegular, pred.XRegular, ...
                                       thisModel.Scale );

    % compute the mean squared error as a function of time
    eval.ReconTimeMSE = reconTemporalLoss( pred.XHatRegular, pred.XRegular, ...
                                           thisModel.Scale );

    figure(4);
    hold on;
    for i = 1:thisDataset.XChannels
        plot( thisDataset.TSpan.Regular, eval.ReconTimeMSE(:,i) );
    end

    % compute the auxiliary loss using the model
    [ pred.YHatEnsemble, pred.YHat ] = predictAux( thisModel, pred.ZEnsemble );
    eval.AuxModelLoss = getPropCorrect( pred.YHat, thisDataset.Y );

    if isa( thisModel, 'FullAEModel' )
        
        if any(thisModel.LossFcnTbl.Types == 'Comparator')
            % compute the comparator loss using the comparator network
            [ pred.ComparatorYHat, eval.ComparatorLoss ] = ...
                            predictCompNet( thisModel, thisDataset ); 
        end

        if any(thisModel.LossFcnTbl.Types == 'Auxiliary')
            % compute the auxiliary loss using the network
            [ pred.AuxNetworkYHat, eval.AuxNetworkLoss ] = ...
                            predictAuxNet( thisModel, pred.ZEnsemble, thisDataset.Y );
        end

    else
        pred.ComparatorYHat = [];
        eval.ComparatorLoss = [];
        pred.AuxNetworkYHat = [];
        eval.AuxNetworkLoss = [];

    end


end
