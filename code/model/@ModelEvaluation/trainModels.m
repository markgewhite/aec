function self = trainModels( self, thisDataset, modelSetup )
    % Run the cross-validation training loop
    arguments
        self            ModelEvaluation
        thisDataset     ModelDataset
        modelSetup      struct
    end

    % prepare the bespoke arguments
    try
        argsModel = namedargs2cell( modelSetup.args );
    catch
        argsModel = {};
    end

    % run the cross validation loop
    for k = 1:self.KFolds
    
        disp(['Fold ' num2str(k) '/' num2str(self.KFolds)]);
        
        % set the kth partitions
        thisTrnSet = thisDataset.partition( self.Partitions(:,k) );
        thisValSet = thisDataset.partition( ~self.Partitions(:,k) );
        
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

end