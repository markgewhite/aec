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
                args.MaxBatchSize           double ...
                    {mustBeInteger, mustBePositive} = 100
                args.DoUseGPU               logical = false
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );

            self@ModelTrainer( lossFcnTbl, ...
                               superArgsCell{:} );

            % setup the environment
            if canUseGPU && args.DoUseGPU
                self.ExecutionEnvironment = 'gpu';
                numGPUs = gpuDeviceCount('available');
                delete( gcp('nocreate') );
                self.Pool = parpool( numGPUs );
            else
                self.ExecutionEnvironment = 'cpu';
                delete( gcp('nocreate') );
                self.Pool = gcp;
            end
            self.NumWorkers = self.Pool.NumWorkers;

            % set the minibatch sizes
            self.BatchSize = min( self.BatchSize.*self.NumWorkers, ...
                                  args.MaxBatchSize );

            % determine batch size per worker
            self.WorkerBatchSize = floor(self.BatchSize ...
                            ./repmat( self.NumWorkers, 1, self.NumWorkers));
            remainder = self.BatchSize - sum( self.WorkerBatchSize );
            self.WorkerBatchSize = self.WorkerBatchSize ...
                + [ones(1, remainder) zeros(1, self.NumWorkers-remainder)];

            % report setup
            disp(['Execution environment = ' self.ExecutionEnvironment]);
            disp(['Worker batch sizes = ' num2str(self.WorkerBatchSize)]);

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