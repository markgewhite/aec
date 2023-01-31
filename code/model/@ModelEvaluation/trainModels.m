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
    
        if isfield( modelSetup.args, 'ShowPlots' )
            if modelSetup.args.ShowPlots
                disp(['Fold ' num2str(k) '/' num2str(self.NumModels)]);
            end
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
        self.Models{k} = modelSetup.class( thisTrnSet, argsModel{:} );

        if self.RandomSeedResets && ~isempty( self.RandomSeed )
            % reset the random seed for the model
            rng( self.RandomSeed );
        end

        % train the model and time it
        tStart = tic;
        self.Models{k} = self.Models{k}.train( thisTrnSet );
        self.Models{k}.Timing.Training.TotalTime = toc(tStart);

        % evaluate the model
        tStart = tic;
        self.Models{k} = self.Models{k}.evaluate( thisTrnSet, thisValSet );
        self.Models{k}.Timing.Testing.TotalTime = toc(tStart);

        % generate the model plots
        if self.Models{k}.ShowPlots
            self.Models{k}.showAllPlots;
        end

        % save the model
        self.Models{k}.save;

    end

    % find the optimal arrangement of model components
    if self.NumModels > 1
        self = self.arrangeComponents;
    end

    % average the latent components across the models
    self.CVComponents = self.calcCVComponents;

    % average the auxiliary model coefficients
    self.CVAuxMetrics.AuxModelBeta = calcCVNestedParameter( ...
                                        self.Models, {'AuxModel', 'Beta'} );


end