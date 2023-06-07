function reportProgress( thisModel, dlZ, dlY, lossTrn, lossVal, epoch )
    % Report progress on training
    arguments
        thisModel       AEModel
        dlZ             dlarray
        dlY             dlarray
        lossTrn         double
        lossVal         double
        epoch           double
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

    % generate the centred components
    XC = thisModel.calcLatentComponents( dlZ, convert = true ) ;
    plotLatentComp( thisModel, ...
                    XC = XC, ...
                    smooth = false, ...
                    centredYAxis = thisModel.HasCentredDecoder );

    % plot the Z distributions
    plotZDist( thisModel, dlZ );

    % plot the Z clusters
    plotZClusters( thisModel, dlZ, Y = dlY );

    drawnow;
      
end