function reportProgress( thisModel, dlZ, dlY, lossTrn, epoch, args )
    % Report progress on training
    arguments
        thisModel       SubAEModel
        dlZ             dlarray
        dlY             dlarray
        lossTrn         double
        epoch           double
        args.nLines     double = 8
        args.lossVal    double = []
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
    if ~isempty( args.lossVal )
        fprintf(' : %1.3f', args.lossVal );
    else
        fprintf('        ');
    end
    fprintf('\n');

    % compute the AE components for plotting
    [ dlXC, dlXMean ] = thisModel.calcLatentComponents( dlZ );

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
    thisModel.AuxModel = trainAuxModel( thisModel.AuxModelType, dlZ, dlY );
    [auxALE, Q] = thisModel.calcALE( dlZ, ...
                                  modelFcn = @predictAuxModel, ...
                                  maxObs = 500 );
    plotALE( thisModel, quantiles = Q, pts = auxALE, type = 'Model' );

    % compute the ALE curves for the auxiliary network, if present
    if any(thisModel.LossFcnTbl.Types == 'Auxiliary')
        [auxALE, Q] = thisModel.calcALE( dlZ, ...
                                      modelFcn = @predictAuxNet, ...
                                      maxObs = 500 );
        plotALE( thisModel, quantiles = Q, pts = auxALE, type = 'Network' );

    end
    drawnow;
      
end