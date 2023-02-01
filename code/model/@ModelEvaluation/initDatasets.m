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

    disp('Loading the data');
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
            self.KFolds = [];
            self.KFoldRepeats = [];
            self.NumModels = 1;

        case 'KFold'
            self.TrainingDataset = setup.data.class( 'Combined', ...
                                            argsCell{:} );

            self.Partitions = self.TrainingDataset.getCVPartition( ...
                                    KFolds = self.KFolds, ...
                                    Repeats = self.KFoldRepeats, ...
                                    Identical = self.HasIdenticalPartitions );
            self.NumModels = size( self.Partitions, 2 );
            
        otherwise
            eid = 'evaluation:UnrecognisedType';
            msg = 'Unrecognised EvaluationType.';
            throwAsCaller( MException(eid,msg) );

    end


end