function reportProgress( thisModel, dlZ, dlY, lossTrn, lossVal, epoch, args )
    % Report progress on training
    arguments
        thisModel       AEModel
        dlZ             dlarray
        dlY             dlarray
        lossTrn         double
        lossVal         double
        epoch           double
        args.nLines     double = 8
    end

    if size( lossTrn, 1 ) > 1
        meanLoss = mean( lossTrn );
    else
        meanLoss = lossTrn;
    end

    fprintf('Loss (%4d) = ', epoch);
    for k = 1:thisModel.NumLoss
        fprintf(' %6.3f', meanLoss(k) );
    end
    if ~isempty( lossVal )
        fprintf(' : %1.3f', lossVal );
    else
        fprintf('        ');
    end
    fprintf('\n');

    % compute the AE components for plotting
    [ dlXC, dlXMean ] = thisModel.calcLatentComponents( dlZ, smooth = true );

    % plot them on specified axes
    if thisModel.HasCentredDecoder
        dlXMean(:) = 0;
    end

    plotLatentComp( thisModel, ...
                    XMean = dlXMean, XC = dlXC, ...
                    centredYAxis = thisModel.HasCentredDecoder, ...
                    type = 'Smoothed', shading = true );

    % plot the Z distributions
    plotZDist( thisModel, dlZ );

    % plot the Z clusters
    plotZClusters( thisModel, dlZ, Y = dlY );

    % fit the auxiliary model and compute the ALE curves
    dlZAux = dlZ( 1:thisModel.ZDimAux, : );
    thisModel.AuxModel = trainAuxModel( thisModel.AuxModelType, dlZAux, dlY );
    [auxALE, Q] = thisModel.calcResponse( dlZ, ...
                                  modelFcn = @predictAuxModel, ...
                                  maxObs = 500 );
    plotAuxResponse( thisModel, quantiles = Q, pts = auxALE, type = 'Model' );

    % compute the ALE curves for the auxiliary network, if present
    if any(thisModel.LossFcnTbl.Types == 'Auxiliary')
        [auxALE, Q] = thisModel.calcALE( dlZ, ...
                                      modelFcn = @predictAuxNet, ...
                                      maxObs = 500 );
        plotAuxResponse( thisModel, quantiles = Q, pts = auxALE, type = 'Network' );

    end
    drawnow;
      
end