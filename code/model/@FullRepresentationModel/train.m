function self = train( self, thisDataset )
    % Train the model on the data provided using cross validation
    arguments
        self            FullRepresentationModel
        thisDataset     ModelDataset
    end

    if ~isempty( self.RandomSeed )
        % set random seed for reproducibility
        rng( self.RandomSeed );
    end

    % re-partition the data to create training and validation sets
    self.Partitions = thisDataset.getCVPartition( ...
                                KFold = self.KFolds, ...
                                Identical = self.IdenticalPartitions );
    
    % run the cross validation loop
    for k = 1:self.KFolds
    
        if self.ShowPlots
            disp(['Fold ' num2str(k) '/' num2str(self.KFolds)]);
        end
        
        % set the kth partitions
        thisTrnSet = thisDataset.partition( self.Partitions(:,k) );
        thisValSet = thisDataset.partition( ~self.Partitions(:,k) );
        
        % initialize the sub-model
        self = self.initModel( k );

        if self.RandomSeedResets && ~isempty( self.RandomSeed )
            % reset the random seed for the submodel
            rng( self.RandomSeed );
        end

        % train the sub-model
        self.SubModels{k} = self.SubModels{k}.train( thisTrnSet );

        % evaluate the sub-model
        self.SubModels{k} = self.SubModels{k}.evaluate( ...
                                    thisTrnSet, thisValSet );

        % save the model
        self.SubModels{k}.save;

    end

    % set the overall mean curve
    self.MeanCurve = thisDataset.XInputRegularMean;

    % find the optimal arrangement of sub-model components
    if self.KFolds > 1
        % NEED TO FIX
        % self = self.arrangeComponents;
    end

    % calculate the aggregated latent components
    self = self.setLatentComponents;

    % calculate the cross-validated losses
    self = self.computeCVLosses;

    % calculate cross-validated correlation matrices
    self = self.computeCVCorrelations;

    % calculate other cross-validated parameters
    self = self.computeCVParameters;

    if self.ShowPlots
        self.plotAllLatentComponents;
        self.plotAllALE;
    end
    
    % save the full model
    self.save;

end
