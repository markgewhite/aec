function lossVal = validationCheck( thisModel, valType, dlXVal, dlXNVal, dlYVal )
    % Validate the model so far
    arguments
        thisModel       AEModel
        valType         string
        dlXVal          dlarray
        dlXNVal         dlarray
        dlYVal          dlarray
    end

    dlZVal = thisModel.encode( dlXVal, convert = false );

    switch valType
        case 'Reconstruction'
            dlXValHat = squeeze(thisModel.reconstruct( dlZVal, convert = false ));
            lossVal = reconLoss( dlXNVal, dlXValHat, thisModel.Scale );

        case 'AuxNetwork'
            dlYHatVal = predict( thisModel.Nets.(thisModel.auxNetwork), dlZVal );
            cLabels = thisModel.LossFcns.(thisModel.auxNetwork).CLabels;
            dlYHatVal = double( onehotdecode( dlYHatVal, single(cLabels), 1 ));
            lossVal = sum( dlYHatVal~=dlYVal )/length(dlYVal);
    end


end