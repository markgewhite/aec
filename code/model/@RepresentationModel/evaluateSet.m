function [eval, pred, cor] = evaluateSet( thisModel, thisDataset )
    % Evaluate the model with a specified dataset
    arguments
        thisModel       RepresentationModel
        thisDataset     ModelDataset
    end

    % record the input
    pred.XInput = squeeze( thisDataset.XInput );
    pred.XTarget = squeeze( thisDataset.XTarget(thisModel.TSpan.Target) );
    pred.Y = thisDataset.Y;

    % generate latent encoding using the trained model
    pred.Z = thisModel.encode( thisDataset );
    pred.ZAux = thisModel.encode( thisDataset, auxiliary = true );

    % reconstruct the curves
    [ pred.XHat, pred.XHatSmoothed ] = thisModel.reconstruct( pred.Z, smooth = true );
       
    % compute reconstruction loss
    eval.ReconLoss = reconLoss( pred.XTarget, pred.XHat, thisModel.Scale );
    eval.ReconLossSmoothed = reconLoss( pred.XHatSmoothed, pred.XInput, ...
                                        thisModel.Scale );

    % compute reconstruction roughness
    eval.ReconRoughness = reconRoughnessLoss( pred.XHat, thisModel.Scale );
    
    % compute the bias and variance
    eval.ReconBias = reconBias( pred.XTarget, pred.XHat, thisModel.Scale );
    eval.ReconVar = eval.ReconLoss - eval.ReconBias^2;
    eval.ReconBiasSmoothed = reconBias( pred.XInput, pred.XHatSmoothed, thisModel.Scale );
    eval.ReconVarSmoothed = eval.ReconLossSmoothed - eval.ReconBiasSmoothed^2;    
    
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
    eval.ReconTimeVar = reconTemporalLoss( XDiff, pred.XTarget, thisModel.Scale );

    % compute the mean squared error as a function of time
    eval.ReconTimeMSESmoothed = reconTemporalLoss( pred.XHatSmoothed, pred.XInput, ...
                                           thisModel.Scale );

    % compute the mean error (bias) as a function of time
    eval.ReconTimeBiasSmoothed = reconTemporalBias( pred.XHatSmoothed, pred.XInput, ...
                                           thisModel.Scale );

    % compute the variance as a function of time
    if length( size(pred.XHatSmoothed) ) == 2
        XDiff = pred.XHatSmoothed - eval.ReconTimeBiasSmoothed;
    else
        XDiff = pred.XHatSmoothed - reshape( eval.ReconTimeBiasSmoothed, ...
                                    size(eval.ReconTimeBiasSmoothed,1), ...
                                    1, [] );
    end
    eval.ReconTimeVarSmoothed = reconTemporalLoss( XDiff, ...
                                    pred.XInput, thisModel.Scale );

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

    pred.AuxModelYHat = predictAuxModel( thisModel, ZLong );
    switch thisModel.AuxObjective
        case 'Classification'
            eval.AuxModel = evaluateClassifier( pred.Y, pred.AuxModelYHat );
        case 'Regression'
            eval.AuxModel = evaluateRegressor( pred.Y, pred.AuxModelYHat );
    end

    % store the model coefficients - all important
    switch class( thisModel.AuxModel )
        case 'ClassificationLinear'
            eval.AuxModel.Coeff = thisModel.AuxModel.Beta;
        case 'ClassificationDiscriminant'
            eval.AuxModel.Coeff = thisModel.AuxModel.DeltaPredictior;
        case 'RegressionLinear'
            eval.AuxModel.Coeff = thisModel.AuxModel.Beta;
    end

    % extract the two largest coefficients
    sortedCoeff = sort( abs(eval.AuxModel.Coeff), 'descend' );
    eval.AuxModel.Coeff1st = sortedCoeff(1);
    if length(sortedCoeff)>1
        eval.AuxModel.Coeff2nd = sortedCoeff(2);
        eval.AuxModel.CoeffRatio = sortedCoeff(1)/sortedCoeff(2);
    else
        eval.AuxModel.Coeff2nd = NaN;
        eval.AuxModel.CoeffRatio = NaN;
    end
    
    % flatten the structure for summarising later
    eval = flattenStruct( eval );

end
