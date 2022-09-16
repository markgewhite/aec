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

    lossVal = 0;
    if any(strcmp( valType, {'Reconstruction', 'Both'} ))
        dlXValHat = squeeze(thisModel.reconstruct( dlZVal, convert = false ));
        lossVal = lossVal + reconLoss( dlXNVal, dlXValHat, thisModel.Scale );
    end

    if any(strcmp( valType, {'AuxNetwork', 'Both'} ))
        dlYHatVal = thisModel.predictAuxNet( dlZVal, convert = false );
        auxNetwork = thisModel.LossFcnTbl.Names(thisModel.LossFcnTbl.Types=="Auxiliary");
        cLabels = thisModel.LossFcns.(auxNetwork).CLabels;
        dlYVal = dlarray( onehotencode( cLabels(dlYVal), 1 ), 'CB' );
        lossVal = lossVal + 0.1*crossentropy( dlYHatVal, dlYVal, ...
                                  'TargetCategories', 'Independent' );
    end

end