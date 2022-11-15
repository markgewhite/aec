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

    % initialize counters
    nIter = 2; % iterationsPerEpoch( mbqTrn );
    i = 0;
    v = 0;
    vp = self.ValPatience;

    % initialize logs
    nTrnLogs = nIter*self.NumEpochs;
    nValLogs = max( ceil( (self.NumEpochs-self.NumEpochsPreTrn) ...
                                /self.ValFreq ), 1 );
    self.LossTrn = zeros( nTrnLogs, thisModel.NumLoss );
    self.LossVal = zeros( nValLogs, 1 );

    nMetricLogs = max( nTrnLogs/(nIter*self.UpdateFreq), 1 );
    self.Metrics = table( ...
        zeros( nMetricLogs, 1 ), ...
        zeros( nMetricLogs, 1 ), ...
        zeros( nMetricLogs, 1 ), ...
        zeros( nMetricLogs, 1 ), ...
        VariableNames = {'ZCorrelation', 'XCCorrelation', ...
                         'ZCovariance', 'XCCovariance'} );

    % initialize the local timers
    thisModel.Timing.Training.ValCheckTime = 0;
    thisModel.Timing.Training.ReportingTime = 0;
   
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

    i = 0;
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
            self.CurrentEpoch = epoch;
    
            % Pre-training
            self.PreTraining = (epoch<=self.NumEpochsPreTrn);
    
            thisModel.LossFcnTbl.DoCalcLoss( thisModel.LossFcnTbl.Types=="Reconstruction" ) ...
                = ~self.PreTraining;
        
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
                                                        thisModel.Nets, ...
                                                        thisModel, ...
                                                        wkXTTrn, ...
                                                        wkXNTrn, ...
                                                        wkPTrn, ...
                                                        wkYTrn, ...
                                                        self.PreTraining );

                % aggregate the losses across all workers
                wkNormFactor = self.WorkerBatchSize(spmdIndex)./self.BatchSize;
                lossTrn = spmdPlus( wkNormFactor*extractdata(wkLossTrn) );
                self.LossTrn(i,1+self.PreTraining:end) = lossTrn;
    
                % aggregate the network states and gradients across all workers
                for m = 1:thisModel.NumNetworks
                    thisName = thisModel.NetNames{m};
                    if isfield( wkStates, thisName )
                        thisModel.Nets.(thisName).State = ...
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
                thisModel.Nets  = thisModel.Optimizer.updateNets( ...
                                        thisModel.Nets, wkGrads, i );   

            end

            % stop training if the Stop button has been clicked
            stopRequest = spmdPlus( stopTrainingEventQueue.QueueLength );

            if spmdIndex == 1
                % send monitoring information to the client
  
                if self.ShowPlots
                    % update loss plots
                    data = [i, self.LossTrn(i,:)];
                    send( dataQueueLoss, gather(data) );
                end

                if ~self.PreTraining ...
                        && mod( epoch, self.ValFreq )==0 ...
                        && self.Holdout > 0
                    
                    % run a validation check
                    v = v + 1;
                    tic;
                    self.LossVal(v) = validationFcn( thisModel );
                    thisModel.Timing.Training.ValCheckTime = ...
                                    thisModel.Timing.Training.ValCheckTime + toc;
        
                    if v > 2*vp-1
                        if min(self.LossVal(1:v)) ...
                                < min(self.LossVal(v-vp+1:v))
                            disp(['Stopping criterion met. Epoch = ' num2str(epoch)]);
                            break
                        end
                    end
    
                end
        
                % update progress on screen
                if mod( epoch, self.UpdateFreq )==0 && self.ShowPlots
                    
                    tic
                    if ~self.PreTraining && self.Holdout > 0 && v > 0
                        % include validation
                        lossVal = self.LossVal( v );
                    else
                        % exclude validation
                        lossVal = [];
                    end
        
                    % record relevant metrics
                    [ metrics( epoch/self.UpdateFreq, : ), ...
                        dlZTrnAll ] = metricsFcn( thisModel );
        
                    % report 
                    reportFcn( thisModel, ...
                               dlZTrnAll, ...
                               self.LossTrn( i-nIter+1:i, : ), ...
                               lossVal, ...
                               epoch );
                    thisModel.Timing.Training.ReportingTime = ...
                        thisModel.Timing.Training.ReportingTime + toc;
        
                end
            
                % update the number of dimensions, if required
                if mod( epoch, self.ActiveZFreq )==0
                    thisModel = thisModel.incrementActiveZDim;
                end
        
                if mod( epoch, self.LRFreq )==0
                    % update learning rates
                    thisModel.Optimizer = ...
                        thisModel.Optimizer.updateLearningRates( self.PreTraining );
                end
            end

        end

    end

end