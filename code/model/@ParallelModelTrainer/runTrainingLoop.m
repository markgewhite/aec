function thisModel = runTrainingLoop( self, ...
                                      thisModel, ...
                                      dsTrn, ...
                                      preprocFcn, ...
                                      lossLinesFcn, ...
                                      validationFcn, ...
                                      metricsFcn, ...
                                      reportFcn, ...
                                      isFixedLength )
    % Carry out the custom training loop using parallel processing
    arguments
        self            ParallelModelTrainer
        thisModel       AEModel
        dsTrn
        preprocFcn      function_handle
        lossLinesFcn    function_handle
        validationFcn   function_handle
        metricsFcn      function_handle
        reportFcn       function_handle
        isFixedLength   logical
    end
  
    % find indices of the mean and variance state parameters 
    % of the batch normalization layers in the network state property.
    % so that mean and variance can be aggregated across all workers
    hasStateLayer = batchNormalizationStateLayers( thisModel.Nets );
    
    % create a data queue to allow training to be terminated early, 
    % if user presses the stop button
    spmd
        stopTrainingEventQueue = parallel.pool.DataQueue;
    end
    stopTrainingQueue = stopTrainingEventQueue{1};

    % complete monitoring setup
    dataQueueLoss = parallel.pool.DataQueue;
    dataQueueValidation = parallel.pool.DataQueue;
    dataQueueMetrics = parallel.pool.DataQueue;
    dataQueueReport = parallel.pool.DataQueue;

    afterEach( dataQueueLoss, lossLinesFcn );
    afterEach( dataQueueValidation, validationFcn );
    afterEach( dataQueueMetrics, metricsFcn );
    afterEach( dataQueueReport, reportFcn );

    % initialize logs
    nIter = numpartitions( dsTrn, self.Pool );
    nTrnLogs = nIter*self.NumEpochs;
    nValLogs = max( ceil( (self.NumEpochs-self.NumEpochsPreTrn)*nIter ...
                                /self.ValFreq ), 1 );
    lossTrn = zeros( nTrnLogs, thisModel.NumLoss );
    lossVal = zeros( nValLogs, 1 );

    nMetricLogs = max( ceil(nTrnLogs/self.UpdateFreq), 1 );
    metrics = table( ...
        zeros( nMetricLogs, 1 ), ...
        zeros( nMetricLogs, 1 ), ...
        zeros( nMetricLogs, 1 ), ...
        zeros( nMetricLogs, 1 ), ...
        VariableNames = {'ZCorrelation', 'XCCorrelation', ...
                         'ZCovariance', 'XCCovariance'} );

    % set worker assignments for additional tasks
    lossLineWorker = 1;
    validationWorker = max( 2, self.NumWorkers );
    if self.NumWorkers > 2
        metricsWorker = 3;
    else
        metricsWorker = 1;
    end

    % initialize the local timers
    valCheckTime = 0;
    reportingTime = 0;

    % take copy of networks for copying to workers
    nets = thisModel.Nets;
    optimizer = thisModel.Optimizer;

    i = 0;
    j = 0;
    v = 0;
    vp = self.ValPatience;
    stopRequest = false;
    
    % begin the parallel training - each worker runs this code
    spmd
    %spmdIndex = 1;

        % partition mini-batch queue to divide it up among the workers
        wkDSTrn = partition( dsTrn, self.NumWorkers, spmdIndex );
    
        % create minibatchqueue using partitioned datastore on each worker
        wkMbqTrn = minibatchqueue( wkDSTrn, 4, ...
                      MiniBatchSize = self.WorkerBatchSize(spmdIndex), ...
                      PartialMiniBatch = self.PartialBatch, ...
                      MiniBatchFcn = preprocFcn, ...
                      MiniBatchFormat = {thisModel.XDimLabels, ...
                                         thisModel.XNDimLabels, 'CB', 'CB'} );
    
        epoch = 0;

        while epoch < self.NumEpochs && ~stopRequest

            epoch = epoch + 1;
            
            % Pre-training
            preTraining = (epoch<=self.NumEpochsPreTrn);

            if isFixedLength && self.HasMiniBatchShuffle
                
                % reset with a shuffled order
                if self.HasShuffleRandomStream
                    % switch random streams for shuffling
                    modelRandomState = rng;
                    if epoch > 1
                        rng( shuffleRandomState );
                    end
                end
    
                shuffle( wkMbqTrn );
                
                if self.HasShuffleRandomStream
                    % switch back to the model random stream
                    shuffleRandomState = rng;
                    rng( modelRandomState );  
                end
            
            else
            
                % reset whilst preserving the order
                reset( wkMbqTrn );
            
            end
        
            % loop over mini-batches
            while spmdReduce(@and, hasdata(wkMbqTrn)) && ~stopRequest         
                i = i + 1;
                
                % read mini-batch of data
                [ wkXTTrn, wkXNTrn, wkPTrn, wkYTrn ] = next( wkMbqTrn );
                
                % evaluate the model gradients 
                [ wkGrads, wkStates, wkLossTrn ] = ...
                                              dlfeval(  @self.gradients, ...
                                                        nets, ...
                                                        thisModel, ...
                                                        wkXTTrn, ...
                                                        wkXNTrn, ...
                                                        wkPTrn, ...
                                                        wkYTrn, ...
                                                        preTraining );

                % aggregate the losses across all workers
                wkNormFactor = self.WorkerBatchSize(spmdIndex)./self.BatchSize;
                lossTrn( i, 1+preTraining:end ) = spmdPlus( wkNormFactor*wkLossTrn );
    
                % aggregate the network states and gradients across all workers
                for m = 1:thisModel.NumNetworks
                    thisName = thisModel.NetNames{m};
                    if isfield( wkStates, thisName )
                        nets.(thisName).State = ...
                                    aggregateState( wkStates.(thisName), ...
                                                    wkNormFactor, ...
                                                    hasStateLayer.(thisName) );
                        wkGrads.(thisName).Value = ...
                                    dlupdate( @aggregateGradients, ...
                                              wkGrads.(thisName).Value, ...
                                              {wkNormFactor} );
                    end
                end               
    
                % update network parameters
                nets  = thisModel.Optimizer.updateNets( nets, wkGrads, i );
        
                if mod( i, self.LRFreq )==0
                    % update learning rates
                    optimizer = optimizer.updateLearningRates( preTraining );
                end

                % stop training if the Stop button has been clicked
                stopRequest = spmdPlus( stopTrainingEventQueue.QueueLength );
           
                if spmdIndex == lossLineWorker
    
                    % update progress on screen
                    if mod( i, self.UpdateFreq )==0 && self.ShowPlots                   
                        % update loss plots (no other plots to save time)
                        tic
                        fcnData = [i, lossTrn(i,:)];
                        send( dataQueueLoss, gather(fcnData) );
                        reportingTime = reportingTime + toc;
                    end
    
                end
    
                if spmdIndex == metricsWorker
                    if mod( i, self.UpdateFreq )==0 && self.ShowPlots                   
                        % record relevant metrics
                        tic;
                        j = j + 1;
                        metrics( j, : ) = metricsFcn( thisModel );
                        reportingTime = reportingTime + toc;
                    end
                end
                
                if spmdIndex == validationWorker
     
                    if ~self.PreTraining ...
                            && mod( i, self.ValFreq )==0 ...
                            && self.Holdout > 0
                        
                        % run a validation check
                        v = v + 1;
                        tic;
                        lossVal(v) = validationFcn( thisModel );
                        valCheckTime = valCheckTime + toc;
            
                        if v > 2*vp-1
                            if min(lossVal(1:v)) < min(lossVal(v-vp+1:v))
                                disp(['Stopping criterion met. Epoch = ' num2str(epoch)]);
                                stopRequest = true;
                            end
                        end
        
                    end
    
                end
    
            end

        end

    end

    % update the trainer using the first worker's logs
    self.LossTrn = lossTrn{1};
    % trim back logs to actual length
    self.LossTrn = self.LossTrn( 1:i{1}, : );
    self.LossVal = lossVal{validationWorker};
    self.LossVal = self.LossVal(1:v{validationWorker});
    self.Metrics = metrics{metricsWorker};
    self.Metrics = self.Metrics(1:j{metricsWorker},:);

    % update the model in the same way
    thisModel.Nets = nets{1};
    thisModel.Optimizer = optimizer{1};

    thisModel.Timing.Training.ValCheckTime = valCheckTime{validationWorker};
    thisModel.Timing.Training.ReportingTime = ...
            reportingTime{lossLineWorker} + reportingTime{metricsWorker};

    % update trainer with final position
    self.CurrentEpoch = epoch{1};


end