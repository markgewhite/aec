function thisModel = runTraining( self, thisModel, thisDataset )
    % Run the training loop for the model
    arguments
        self            ModelTrainer
        thisModel       AEModel
        thisDataset     ModelDataset
    end

    % re-partition the data to create training and validation sets
    trainObs = thisDataset.getCVPartition( Holdout = self.Holdout );
    
    thisTrnData = thisDataset.partition( trainObs );
    thisValData = thisDataset.partition( ~trainObs );

    % set the mean curve for these training data
    thisModel.MeanCurveTarget = thisTrnData.XTargetMean;
    thisModel.MeanCurve = thisTrnData.XInputRegularMean;

    % setup the minibatch queues
    mbqTrn = getMiniBatchQueue( thisTrnData, ...
                                self.BatchSize, ...
                                thisModel.XDimLabels, ...
                                thisModel.XNDimLabels, ...
                                partialBatch = self.PartialBatch );

    if self.Holdout > 0
        % get the validation data (one-time only)
        [ dlXVal, dlYVal ] = thisValData.getDLInput( thisModel.XDimLabels );
        dlXNVal = thisValData.XTarget;
    end

    % setup whole training set
    [ dlXTrnAll, dlYTrnAll ] = thisTrnData.getDLInput( thisModel.XDimLabels );
   
    % setup monitoring functions
    lossLinesFcn = @(i, l) updateLossLines( self.LossLines, i, l );
    validationFcn = @(model) validationCheck( model, ...
                                              self.ValType, ...
                                              dlXVal, dlXNVal, dlYVal );
    metricsFcn = @(model) calcMetrics( model, dlXTrnAll );
    reportFcn = @(model, z, l1, l2, e) reportProgress( ...
                                    model, z, thisTrnData.Y, l1, l2, e );

    % execute the custom training loop
    thisModel = self.runTrainingLoop( thisModel, ...
                                      mbqTrn, ...
                                      lossLinesFcn, ...
                                      validationFcn, ...
                                      metricsFcn, ...
                                      reportFcn, ...
                                      thisTrnData.isFixedLength );
   

    % train the auxiliary model
    tic;
    dlZTrnAll = thisModel.encode( dlXTrnAll, convert = false );
    dlZTrnAllAux = dlZTrnAll( 1:thisModel.ZDimAux, : );
    [thisModel.AuxModel, ...
        thisModel.AuxModelZMean, ...
        thisModel.AuxModelZStd] = trainAuxModel( ...
                                thisModel.AuxModelType, ...
                                dlZTrnAllAux, ...
                                dlYTrnAll );
    thisModel.Timing.Training.AuxModelTime = toc;

    % set the oversmoothing level
    XHatTrnAll = thisModel.reconstruct( dlZTrnAll, convert = true );

    [ thisModel.FDA.FdParamsTarget, thisModel.FDA.LambdaTarget ] = ...
        thisTrnData.setFDAParameters( thisTrnData.TSpan.Target, ...
                                      permute(XHatTrnAll, [1 3 2]) );

end
    