function thisModel = runTraining( self, thisModel, thisDataset )
    % Run the training loop for the model
    arguments
        self            ModelTrainer
        thisModel       SubAEModel
        thisDataset     ModelDataset
    end

    % re-partition the data to create training and validation sets
    trainObs = thisDataset.getCVPartition( Holdout = self.Holdout );
    
    thisTrnData = thisDataset.partition( trainObs );
    thisValData = thisDataset.partition( ~trainObs );

    % setup the minibatch queues
    mbqTrn = thisTrnData.getMiniBatchQueue( ...
                                self.BatchSize, ...
                                thisModel.XDimLabels, ...
                                thisModel.XNDimLabels, ...
                                partialBatch = self.PartialBatch );

    if self.Holdout > 0
        % get the validation data (one-time only)
        [ dlXVal, dlYVal ] = thisValData.getDLInput( thisModel.XDimLabels );
    end

    % setup whole training set
    [ dlXTrnAll, dlYTrnAll ] = thisTrnData.getDLInput( thisModel.XDimLabels );

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

    for epoch = 1:self.NumEpochs
        
        self.CurrentEpoch = epoch;

        % Pre-training
        self.PreTraining = (epoch<=self.NumEpochsPreTrn);

        thisModel.LossFcnTbl.DoCalcLoss( thisModel.LossFcnTbl.Types=="Reconstruction" ) ...
            = ~self.PreTraining;
    
        if thisTrnData.isFixedLength && self.HasMiniBatchShuffle
            
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
            [ dlXTTrn, dlXNTrn, dlYTrn ] = next( mbqTrn );
            
            % evaluate the model gradients 
            [ grads, states, self.LossTrn(j,1+self.PreTraining:end) ] = ...
                              dlfeval(  @gradients, ...
                                        thisModel.Nets, ...
                                        thisModel, ...
                                        dlXTTrn, ...
                                        dlXNTrn, ...
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
                updateLossLines( self.LossLines, j, self.LossTrn(j,:) );
            end

        end
                       
        if ~self.PreTraining ...
                && mod( epoch, self.ValFreq )==0 ...
                && self.Holdout > 0
            
            % run a validation check
            v = v + 1;
           
            % compute relevant loss
            self.LossVal(v) = validationCheck( thisModel, ...
                                            self.ValType, ...
                                            dlXVal, dlYVal );
            if v > 2*vp-1
                if mean(self.LossVal(v-2*vp+1:v-vp)) ...
                        < mean(self.LossVal(v-vp+1:v))
                    disp(['Stopping criterion met. Epoch = ' num2str(epoch)]);
                    break
                end
            end

        end
    
        % update progress on screen
        if mod( epoch, self.UpdateFreq )==0 && self.ShowPlots
            
            if ~self.PreTraining && self.Holdout > 0 && v > 0
                % include validation
                lossValArg = self.LossVal( v );
            else
                % exclude validation
                lossValArg = [];
            end

            % record relevant metrics
            [ self.Metrics( epoch/self.UpdateFreq, : ), ...
                dlZTrnAll ] = calcMetrics( thisModel, dlXTrnAll );

            % report 
            reportProgress( thisModel, ...
                            dlZTrnAll, ...
                            thisTrnData.Y, ...
                            self.LossTrn( j-nIter+1:j, : ), ...
                            epoch, ...
                            lossVal = lossValArg );
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
   

    % train the auxiliary model
    dlZTrnAll = thisModel.encode( dlXTrnAll, convert = false );
    [thisModel.AuxModel, ...
        thisModel.AuxModelZMean, ...
        thisModel.AuxModelZStd] = trainAuxModel( ...
                                thisModel.AuxModelType, ...
                                dlZTrnAll, ...
                                dlYTrnAll );

    % set the mean curve for these training data
    thisModel.MeanCurveTarget = thisTrnData.XTargetMean;
    thisModel.MeanCurve = thisTrnData.XInputRegularMean;

    % set the oversmoothing level
    XHatTrnAll = thisModel.reconstruct( dlZTrnAll, convert = true );
    [ thisModel.FDA.FdParamsTarget, thisModel.FDA.LambdaTarget ] = ...
        thisTrnData.setTargetFdParams( permute(XHatTrnAll, [1 3 2]) );

end
    