function self = trainModels( self, modelSetup )
    % Run the cross-validation training loop
    arguments
        self                ModelEvaluation
        modelSetup          struct
    end

    % prepare the bespoke arguments
    try
        argsModel = namedargs2cell( modelSetup.args );
    catch
        argsModel = {};
    end

    % run the cross validation loop
    for k = 1:self.NumModels
    
        if self.Verbose
            disp(['Fold ' num2str(k) '/' num2str(self.NumModels)]);
        end
        
        switch self.CVType
            case 'Holdout'
                % set the training and holdout data sets
                thisTrnSet = self.TrainingDataset;
                thisValSet = self.TestingDataset;
            case 'KFold'
                % set the kth partitions
                thisTrnSet = self.TrainingDataset.partition( self.Partitions(:,k) );
                thisValSet = self.TrainingDataset.partition( ~self.Partitions(:,k) );
        end
        
        % initialize the model
        if self.NumModels > 1
            foldName = [self.Name '-Fold' num2str( k, '%02d' )];
        else
            foldName = self.Name;
        end
        self.Models{k} = modelSetup.class( thisTrnSet, ...
                                           argsModel{:}, ...
                                           Name = foldName );

        if self.RandomSeedResets && ~isempty( self.RandomSeed )
            % reset the random seed for the model
            rng( self.RandomSeed );
        end

        % train the model and time it
        if self.Verbose
            disp('Training the model...');
        end
        tStart = tic;
        self.Models{k} = self.Models{k}.train( thisTrnSet );
        self.Models{k}.Timing.Training.TotalTime = toc(tStart);

        % evaluate the model
        if self.Verbose
            disp('Evaluating the model');
        end
        tStart = tic;
        self.Models{k} = self.Models{k}.evaluate( thisTrnSet, thisValSet );
        self.Models{k}.Timing.Testing.TotalTime = toc(tStart);

        % generate the model plots and save them
        self.Models{k}.showAllPlots;
        %self.Models{k}.save;

    end

    % find the optimal arrangement of model components
    if self.NumModels > 1
        if self.Verbose
            disp('Aligning components...');
        end
        self = self.arrangeComponents;
    end

    % average the latent components across the models
    self.CVComponents = self.calcCVComponents;

    % average the auxiliary model coefficients
    self.CVAuxMetrics.AuxModelBeta = calcCVNestedParameter( ...
                                        self.Models, {'AuxModel', 'Beta'} );


end