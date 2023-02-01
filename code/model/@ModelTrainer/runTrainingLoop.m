function thisModel = runTrainingLoop( self, ...
                                      thisModel, ...
                                      dsTrn, ...
                                      preprocFcn, ...
                                      lossLinesFcn, ...
                                      validationFcn, ...
                                      metricsFcn, ...
                                      reportFcn, ...
                                      isFixedLength )
    % Carry out the custom training loop
    arguments
        self            ModelTrainer
        thisModel       AEModel
        dsTrn
        preprocFcn      function_handle
        lossLinesFcn    function_handle
        validationFcn   function_handle
        metricsFcn      function_handle
        reportFcn       function_handle
        isFixedLength   logical
    end

    % create the mini batch queue
    mbqTrn = minibatchqueue(  dsTrn, 4, ...
                      MiniBatchSize = self.BatchSize, ...
                      PartialMiniBatch = self.PartialBatch, ...
                      MiniBatchFcn = preprocFcn, ...
                      MiniBatchFormat = {thisModel.XDimLabels, ...
                                         thisModel.XNDimLabels, 'CB', 'CB'} );

    % initialize counters
    nIter = self.iterationsPerEpoch( mbqTrn );
    i = 0;
    v = 0;
    vp = self.ValPatience;
    epoch = 0;
    stopping = false;

    % initialize logs
    nTrnLogs = self.NumIterations;
    nValLogs = max( ceil( (self.NumIterations-self.NumIterPreTrn) ...
                                /self.ValFreq ), 1 );
    self.LossTrn = zeros( nTrnLogs, thisModel.NumLoss );
    self.LossVal = zeros( nValLogs, 1 );

    nMetricLogs = max( ceil(nTrnLogs/self.UpdateFreq), 1 );
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
    
    while i < self.NumIterations && ~stopping
        
        epoch = epoch + 1;
        self.CurrentEpoch = epoch;
    
        if isFixedLength && self.HasMiniBatchShuffle
            
            % reset with a shuffled order
            if self.HasShuffleRandomStream
                % switch random streams for shuffling
                modelRandomState = rng;
                if epoch > 1
                    rng( shuffleRandomState );
                end
            end

            shuffle( mbqTrn );
            
            if self.HasShuffleRandomStream
                % switch back to the model random stream
                shuffleRandomState = rng;
                rng( modelRandomState );  
            end
        
        else
        
            % reset whilst preserving the order
            reset( mbqTrn );
        
        end
    
        % loop over mini-batches
        while hasdata( mbqTrn ) && ~stopping
            
            i = i + 1;
            self.CurrentIteration = i;

            % Pre-training
            preTraining = (i<=self.NumIterPreTrn);
            
            % read mini-batch of data
            [ dlXTTrn, dlXNTrn, dlPTrn, dlYTrn ] = next( mbqTrn );
            
            % evaluate the model gradients 
            [ grads, states, self.LossTrn(i,1+preTraining:end) ] = ...
                              dlfeval(  @self.gradients, ...
                                        thisModel.Nets, ...
                                        thisModel, ...
                                        dlXTTrn, ...
                                        dlXNTrn, ...
                                        dlPTrn, ...
                                        dlYTrn, ...
                                        preTraining );

            % store revised network states
            for m = 1:thisModel.NumNetworks
                thisName = thisModel.NetNames{m};
                if isfield( states, thisName )
                    thisModel.Nets.(thisName).State = states.(thisName);
                end
            end

            % update network parameters
            thisModel.Nets  = thisModel.Optimizer.updateNets( ...
                                    thisModel.Nets, grads, i );

            if self.ShowPlots
                % update loss plots
                fcnData = [ i, self.LossTrn(i,:) ];
                lossLinesFcn( fcnData );
            end
                       
            if ~preTraining ...
                    && mod( i, self.ValFreq )==0 ...
                    && self.Holdout > 0
                
                % run a validation check
                v = v + 1;
               
                % compute relevant loss
                tic;
                self.LossVal(v) = validationFcn( thisModel );
                thisModel.Timing.Training.ValCheckTime = ...
                                thisModel.Timing.Training.ValCheckTime + toc;
    
                if v > 2*vp-1
                    if min(self.LossVal(1:v)) ...
                            < min(self.LossVal(v-vp+1:v))
                        disp(['Stopping criterion met. Epoch = ' num2str(epoch)]);
                        stopping = true;
                    end
                end
    
            end
        
            % update progress on screen
            if mod( i, self.UpdateFreq )==0 && self.ShowPlots
                
                tic
                if ~preTraining && self.Holdout > 0 && v > 0
                    % include validation
                    lossVal = self.LossVal( v );
                else
                    % exclude validation
                    lossVal = [];
                end
    
                % record relevant metrics
                [ self.Metrics( i/self.UpdateFreq, : ), ...
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
            if mod( i, self.ActiveZFreq )==0
                thisModel = thisModel.incrementActiveZDim;
            end
    
            if mod( i, self.LRFreq )==0
                % update learning rates
                thisModel.Optimizer = ...
                    thisModel.Optimizer.updateLearningRates( preTraining );
            end

        end

    end

end