function thisModel = runTrainingLoop( self, ...
                                      thisModel, mbqTrn, ...
                                      lossLinesFcn, ...
                                      validationFcn, ...
                                      metricsFcn, ...
                                      reportFcn, ...
                                      isFixedLength )
    % Carry out the custom training loop
    arguments
        self            ModelTrainer
        thisModel       AEModel
        mbqTrn     
        lossLinesFcn    function_handle
        validationFcn   function_handle
        metricsFcn      function_handle
        reportFcn       function_handle
        isFixedLength   logical
    end

    % initialize counters
    nIter = iterationsPerEpoch( mbqTrn );
    j = 0;
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
    
    for epoch = 1:self.NumEpochs
        
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
        for i = 1:nIter
            
            j = j + 1;
            
            % read mini-batch of data
            [ dlXTTrn, dlXNTrn, dlPTrn, dlYTrn ] = next( mbqTrn );
            
            % evaluate the model gradients 
            [ grads, states, self.LossTrn(j,1+self.PreTraining:end) ] = ...
                              dlfeval(  @gradients, ...
                                        thisModel.Nets, ...
                                        thisModel, ...
                                        dlXTTrn, ...
                                        dlXNTrn, ...
                                        dlPTrn, ...
                                        dlYTrn, ...
                                        self.PreTraining );

            % store revised network states
            for m = 1:thisModel.NumNetworks
                thisName = thisModel.NetNames{m};
                if isfield( states, thisName )
                    thisModel.Nets.(thisName).State = states.(thisName);
                end
            end

            % update network parameters
            thisModel.Nets  = thisModel.Optimizer.updateNets( ...
                                    thisModel.Nets, grads, j );

            if self.ShowPlots
                % update loss plots
                lossLinesFcn( j, self.LossTrn(j,:) );
            end

        end
                       
        if ~self.PreTraining ...
                && mod( epoch, self.ValFreq )==0 ...
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
                       self.LossTrn( j-nIter+1:j, : ), ...
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