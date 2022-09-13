function self = initDatasets( self, setup )
    % Initialize the datasets
    arguments
        self        ModelEvaluation
        setup       struct
    end

    try
        argsCell = namedargs2cell( setup.data.args );
    catch
        argsCell = {};
    end

    switch self.CVType
        case 'Holdout'
            self.TrainingDataset = setup.data.class( 'Training', ...
                                            argsCell{:} ); %#ok<*MCNPN> 

            self.TestingDataset = setup.data.class( 'Testing', ...
                                                    argsCell{:}, ...
                    tSpan = self.TrainingDataset.TSpan.Input, ...
                    PaddingLength = self.TrainingDataset.Padding.Length, ...
                    Lambda = self.TrainingDataset.FDA.Lambda );
            self.Partitions = [];
            self.KFolds = 1;
            self.KFoldRepeats = 1;

        case 'KFold'
            self.TrainingDataset = setup.data.class( 'Combined', ...
                                            argsCell{:} );

        otherwise
            eid = 'evaluation:UnrecognisedType';
            msg = 'Unrecognised EvaluationType.';
            throwAsCaller( MException(eid,msg) );

    end

    self.Partitions = self.TrainingDataset.getCVPartition( ...
                            KFolds = self.KFolds, ...
                            Repeats = self.KFoldRepeats, ...
                            Identical = self.HasIdenticalPartitions );
    self.NumModels = size( self.Partitions, 2 );

end