classdef ParallelModelTrainer < ModelTrainer
    % Subclass defining a model trainer using parallel processing

    properties
        ExecutionEnvironment            % type of processing environment
        Pool                            % processing pool
        NumWorkers                      % number of workers
        WorkerBatchSize                 % each worker's batch size
    end

    methods

        function self = ParallelModelTrainer( lossFcnTbl, superArgs, args )
            % Initialize the trainer
            arguments
                lossFcnTbl                  table
                superArgs.?ModelTrainer
                args.DoUseGPU               logical = false
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );

            self@ModelTrainer( lossFcnTbl, ...
                               superArgsCell{:} );

            % setup the environment
            if canUseGPU && args.DoUseGPU
                self.ExecutionEnvironment = "gpu";
                numGPUs = gpuDeviceCount('available');
                delete( gcp('nocreate') );
                self.Pool = parpool( numGPUs );
            else
                self.ExecutionEnvironment = 'cpu';
                self.Pool = gcp;
            end
            self.NumWorkers = self.Pool.NumWorkers;

            % scale-up batch proportional to the number of workers
            self.BatchSize = self.BatchSize.*self.NumWorkers;

            % determine batch size per worker
            self.WorkerBatchSize = floor(self.BatchSize ...
                            ./repmat( self.NumWorkers, 1, self.NumWorkers));
            remainder = self.BatchSize - sum( self.WorkerBatchSize );
            self.WorkerBatchSize = self.WorkerBatchSize ...
                + [ones(1, remainder) zeros(1, self.NumWorkers-remainder)];

        end

        % class methods

        thisModel = runTrainingLoop( self, ...
                                     thisModel, mbqTrn, ...
                                     validationFcn, ...
                                     metricsFcn, ...
                                     reportFcn, ...
                                     isFixedLength )

    end

end