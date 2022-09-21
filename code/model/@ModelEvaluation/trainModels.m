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
    
        disp(['Fold ' num2str(k) '/' num2str(self.NumModels)]);
        
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

        % train the sub-model
        self.Models{k} = self.Models{k}.train( thisTrnSet );

        % evaluate the sub-model
        self.Models{k} = self.Models{k}.evaluate( thisTrnSet, thisValSet );

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