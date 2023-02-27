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
    thisModel.MeanCurve = mean( thisTrnData.XInputRegular, 2 );
    if thisModel.UsesFdCoefficients
        thisModel.MeanCurveTarget = thisTrnData.XTargetCoeffMean;
    else
        thisModel.MeanCurveTarget = thisTrnData.XTargetMean;
    end

    % create a super datastore combining individual variable datastores
    dsTrn = thisTrnData.getDatastore( thisModel.UsesDensityEstimation, ...
                                      thisModel.UsesFdCoefficients );

    % setup the minibatch preprocessing function
    preprocFcn = @( X, XN, P, Y, I ) preprocMiniBatch( X, XN, P, Y, I, ...
                                            thisTrnData.Padding.Value, ...
                                            thisTrnData.Padding.Location );

    if self.Holdout > 0
        % get the validation data (one-time only)
        [ dlXVal, dlYVal, dlXNVal ] = thisModel.getDLArrays( thisValData );
    end

    % setup whole training set
    [ dlXTrnAll, dlYTrnAll ] = thisModel.getDLArrays( thisTrnData );
   
    % setup monitoring functions
    lossLinesFcn = @(data) updateLossLines( self.LossLines, data );
    validationFcn = @(model) validationCheck( model, ...
                                              self.ValType, ...
                                              dlXVal, dlXNVal, dlYVal );
    metricsFcn = @(model) calcMetrics( model, dlXTrnAll );
    reportFcn = @(model, z, l1, l2, e) reportProgress( ...
                                    model, z, thisTrnData.Y, l1, l2, e );

    % execute the custom training loop
    thisModel = self.runTrainingLoop( thisModel, ...
                                      dsTrn, ...
                                      preprocFcn, ...
                                      lossLinesFcn, ...
                                      validationFcn, ...
                                      metricsFcn, ...
                                      reportFcn, ...
                                      thisTrnData.HasNormalizedInput );

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

    % set the smoothing level for the reconstructions
    XHatTrnAll = thisModel.reconstruct( dlZTrnAll );

    [ thisModel.FDA.FdParamsTarget, thisModel.FDA.LambdaTarget ] = ...
        thisTrnData.setFDAParameters( thisTrnData.TSpan.Target, ...
                                        XHatTrnAll );

    % set the smoothing level for the components (may be different)
    XCTrnAll = thisModel.calcLatentComponents( dlZTrnAll, ...
                                               sampling = 'Regular' );
    XCTrnAll = reshape( XCTrnAll, ...
                        thisModel.XComponentDim, [], thisModel.XChannels );
    XCTrnAll = XCTrnAll + reshape( thisTrnData.XTargetMean, ...
                                   thisModel.XComponentDim, 1, thisModel.XChannels );

    [ thisModel.FDA.FdParamsComponent, thisModel.FDA.LambdaComponent ] = ...
        thisTrnData.setFDAParameters( thisTrnData.TSpan.Target, ...
                                      XCTrnAll );


end
    