function self = train( self, thisData )
    % Train the autoencoder
    arguments
        self                    AEModel
        thisData                ModelDataset
    end

    % complete initialization
    self = self.finalizeInit( thisData );

    % perform training using the trainer
    self = self.Trainer.runTraining( self, thisData );

    % train the auxiliary model
    tic;

    [ dlX, dlY ] = self.getDLArrays( thisData );
    dlZ = self.encode( dlX, convert = false );
    dlZAux = dlZ( 1:self.ZDimAux, : );
    [self.AuxModel, self.AuxModelZMean, self.AuxModelZStd] = ...
                                    trainAuxModel( self.AuxModelType, ...
                                                   dlZAux, ...
                                                   dlY );

    self.Timing.Training.AuxModelTime = toc;

    % set the smoothing level for the reconstructions
    XHat = self.reconstruct( dlZ );

    [ self.FDA.FdParamsTarget, self.FDA.LambdaTarget ] = ...
            thisData.setFDAParameters( self.TSpan.Target, XHat );

    % generate the functional components
    self.LatentComponents = self.calcLatentComponents( dlZ, convert = true ); 

    % set the smoothing level for the components (may be different)
    XC = reshape( self.LatentComponents, self.XTargetDim, [], self.XChannels );
    XC = XC + reshape( self.MeanCurveTarget, self.XTargetDim, 1, self.XChannels );

    [ self.FDA.FdParamsComponent, self.FDA.LambdaComponent ] = ...
                        thisData.setFDAParameters( self.TSpan.Target, XC );
    self.FDA.FdParamsComponent = self.FDA.FdParamsInput;
    self.FDA.LambdaComponent = self.FDA.Lambda;


end