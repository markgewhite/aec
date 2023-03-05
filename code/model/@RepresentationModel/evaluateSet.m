function [eval, pred, cor] = evaluateSet( thisModel, thisDataset )
    % Evaluate the model with a specified dataset
    arguments
        thisModel       RepresentationModel
        thisDataset     ModelDataset
    end

    % record the input
    pred.XTarget = squeeze( thisDataset.XTarget );
    pred.XRegular = squeeze( thisDataset.XInputRegular );
    pred.Y = thisDataset.Y;

    % generate latent encoding using the trained model
    pred.Z = thisModel.encode( thisDataset );
    pred.ZAux = pred.Z( :, 1:thisModel.ZDimAux );

    % reconstruct the curves
    [ pred.XHat, pred.XHatSmoothed, pred.XHatRegular ] = ...
            thisModel.reconstruct( pred.Z, smooth = true );
       
    % compute reconstruction loss
    eval.ReconLoss = reconLoss( pred.XTarget, pred.XHat, thisModel.Scale );
    eval.ReconLossSmoothed = reconLoss( pred.XHatSmoothed, pred.XHat, ...
                                        thisModel.Scale );

    % compute reconstruction loss for the regularised curves
    eval.ReconLossRegular = reconLoss( pred.XHatRegular, pred.XRegular, ...
                                       thisModel.Scale );

    % compute reconstruction roughness
    eval.ReconRoughness = reconRoughnessLoss( pred.XHatRegular, ...
                                              thisModel.Scale );
    
    % compute the bias and variance
    eval.ReconBias = reconBias( pred.XTarget, pred.XHat, thisModel.Scale );
    eval.ReconVar = eval.ReconLoss - eval.ReconBias^2;
    eval.ReconBiasRegular = reconBias( pred.XRegular, pred.XHatRegular, thisModel.Scale );
    eval.ReconVarRegular = eval.ReconLossRegular - eval.ReconBiasRegular^2;    
    
    % compute the mean squared error as a function of time
    eval.ReconTimeMSE = reconTemporalLoss( pred.XHat, pred.XTarget, thisModel.Scale );

    % compute the mean error (bias) as a function of time
    eval.ReconTimeBias = reconTemporalBias( pred.XHat, pred.XTarget, thisModel.Scale );

    % compute the variance as a function of time
    if length( size(pred.XHat) ) == 2
        XDiff = pred.XHat - eval.ReconTimeBias;
    else
        XDiff = pred.XHat - reshape( eval.ReconTimeBias, ...
                                    size(eval.ReconTimeBias,1), ...
                                    1, [] );
    end
    eval.ReconTimeVar = reconTemporalLoss( XDiff, pred.XTarget, ...
                                            thisModel.Scale );

    % compute the mean squared error as a function of time
    eval.ReconTimeMSERegular = reconTemporalLoss( pred.XHatRegular, pred.XRegular, ...
                                           thisModel.Scale );

    % compute the mean error (bias) as a function of time
    eval.ReconTimeBiasRegular = reconTemporalBias( pred.XHatRegular, pred.XRegular, ...
                                           thisModel.Scale );

    % compute the variance as a function of time
    if length( size(pred.XHatRegular) ) == 2
        XDiff = pred.XHatRegular - eval.ReconTimeBiasRegular;
    else
        XDiff = pred.XHatRegular - reshape( eval.ReconTimeBiasRegular, ...
                                    size(eval.ReconTimeBiasRegular,1), ...
                                    1, [] );
    end
    eval.ReconTimeVarRegular = reconTemporalLoss( XDiff, ...
                                    pred.XRegular, thisModel.Scale );

    % compute the latent code correlation matrix
    [ cor.ZCorrelation, cor.ZCovariance ] = ...
        latentCodeCorrelation( pred.ZAux, summary = true );
    
    [ cor.ZCorrelationMatrix, cor.ZCovarianceMatrix ] = ...
        latentCodeCorrelation( pred.ZAux );

    % compute the latent component correlation matrix
    [ cor.XCCorrelation, cor.XCCovariance ] = ...
        latentComponentCorrelation( thisModel.LatentComponents,  summary = true );

    [ cor.XCCorrelationMatrix, cor.XCCovarianceMatrix ] = ...
        latentComponentCorrelation( thisModel.LatentComponents );

    cor.XCInnerProduct = innerProductLoss( thisModel.LatentComponents );

    % compute the auxiliary loss using the model
    ZLong = reshape( pred.ZAux, size( pred.ZAux, 1 ), [] );
    ZLong = (ZLong-thisModel.AuxModelZMean)./thisModel.AuxModelZStd;

    pred.AuxModelYHat = predict( thisModel.AuxModel, ZLong );
    eval.AuxModel = evaluateClassifier( pred.Y, pred.AuxModelYHat );

    % store the model coefficients - all important
    switch class( thisModel.AuxModel )
        case 'ClassificationLinear'
            eval.AuxModel.Coeff = thisModel.AuxModel.Beta;
        case 'ClassificationDiscriminant'
            eval.AuxModel.Coeff = thisModel.AuxModel.DeltaPredictior;
    end

    eval = flattenStruct( eval );

end
