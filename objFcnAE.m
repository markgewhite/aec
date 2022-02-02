% ************************************************************************
% Function: objFcnAE
%
% Objective function for the autoencoder to be used for optimisation
%
% Parameters:
%           
% Outputs:
%           setup : initialised setup structure
%
% ************************************************************************

function [ obj, constraint ] = objFcnAE( hyperparams, setup )

% initialise data    
[X, ~, Y, setup.data ] = initializeData( setup.opt.dataSource, ...
                                         setup.opt.nCodes ); 

% initalise autoencoder setup
setup.aae = initializeAE( setup.data );

% update the configuration with the specified settings
setup.aae = unpackHyperparameters( setup.aae, hyperparams );

% impose the same setup for the decoder as the encoder
% setup.aae.dec.dropout = setup.aae.enc.dropout;
setup.aae.dec.filterSize = setup.aae.enc.filterSize;
setup.aae.dec.stride = setup.aae.enc.stride;
setup.aae.dec.nHidden = setup.aae.enc.nHidden;
setup.aae.dec.scale = setup.aae.enc.scale;

% update dependencies
setup.aae.nEpochs = setup.opt.nEpochs;
setup.aae.verbose = false;

% partitioning
rng( setup.opt.randomSeed );
cvPart = cvpartition( Y, 'Holdout', 0.5 );
XTrn = X( :, training(cvPart) );
XTst = X( :, test(cvPart)  );
YTrn = Y( training(cvPart) );
YTst = Y( test(cvPart)  );

% train the autoencoder
[dlnetEnc, dlnetDec, dlnetDis, dlnetCls, lossTrace, constraint ] = ...
                    trainAAE( XTrn, YTrn, setup.aae );
if isnan( lossTrace )
    obj = 0;
    return
end

% switch to DL array format
dlXTrn = dlarray( XTrn, 'CB' );
dlXTst = dlarray( XTst, 'CB' );

% generate encodings
dlZTrn = predict( dlnetEnc, dlXTrn );
dlZTst = predict( dlnetEnc, dlXTst );
if setup.aae.variational
    if setup.aae.useVarMean
        dlZTrn = dlZTrn( 1:setup.aae.zDim, : );
        dlZTst = dlZTst( 1:setup.aae.zDim, : );
    else
        dlZTrn = reparameterize( dlZTrn );
        dlZTst = reparameterize( dlZTst );
    end
end

% convert back to numeric arrays
ZTrn = double(extractdata( dlZTrn ));
ZTst = double(extractdata( dlZTst ));

% classify using discriminant analysis
model = fitcdiscr( ZTrn', YTrn );
errAE = loss( model, ZTst', YTst );

% classify using the trained network
dlYHatTst = predict( dlnetCls, dlZTst );
YHatTst = double(   ...
        onehotdecode( dlYHatTst, single(setup.aae.cLabels), 1 ) )' - 1;
errNet = sum( YHatTst~=YTst )/length(YTst);

% reconstruct the curves and calculate errors
dlXTrnHat = predict( dlnetDec, dlZTrn );
dlXTstHat = predict( dlnetDec, dlZTst );
XTrnHat = double(extractdata( dlXTrnHat ));
XTstHat = double(extractdata( dlXTstHat ));

errTrn = sqrt( mse( XTrn, XTrnHat ) );
errTst = sqrt( mse( XTst, XTstHat ) );

% set the objective function's output
switch setup.opt.objective
    case 'TestError'
        obj = errTst;
    case 'TrainError'
        obj = errTrn;
    case 'OrthogonalLoss'
        obj = mean( lossTrace(end-9:end, 7) );
    case 'ClassificationError'
        obj = errNet;
    otherwise
        error('Unrecognised objective');
end


end

