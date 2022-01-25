% ************************************************************************
% Function: initializeAE
%
% Initialise the autoencoder setup
%
% Parameters:
%           
% Outputs:
%           setup : initialised setup structure
%
% ************************************************************************


function setup = initializeAE( nCodes, classSizes, sigDim )

rng( 'default' );

% data generation parameters
setup.data.nCodes = nCodes;
setup.data.classSizes = classSizes;
setup.data.sigDim = sigDim;
setup.data.tFine = linspace( 0, 1000, 21 );
setup.data.tSpan = linspace( 0, 1024, 33 );
setup.data.ratio = [ 4 8 16];
setup.data.sharedLevel = 3;
setup.data.mu = [1 4 8];
setup.data.sigma = [1 6 1];
setup.data.eta = 0.1;
setup.data.warpLevel = 2;
setup.data.tau = 50;

% functional data analysis parameters
setup.fda.basisOrder = 4;
setup.fda.penaltyOrder = 2;
setup.fda.lambda = 1E2;
setup.fda.nBasis = 20+setup.fda.penaltyOrder+1;
setup.fda.basisFd = create_bspline_basis( ...
                        [ setup.data.tSpan(1), setup.data.tSpan(end) ], ...
                          setup.fda.nBasis, setup.fda.basisOrder);
setup.fda.fdPar = fdPar( setup.fda.basisFd, ...
                         setup.fda.penaltyOrder, ...
                         setup.fda.lambda );
setup.fda.tSpan = setup.data.tFine;

% AAE training parameters
setup.aae.designFcn = @aaeDesign;
setup.aae.gradFcn = @modelGradients;
setup.aae.optimizer = 'ADAM';
setup.aae.nEpochs = 250;
setup.aae.nEpochsPretraining = 10;
setup.aae.batchSize = 100;
setup.aae.beta1 = 0.9;
setup.aae.beta2 = 0.999;
setup.aae.verbose = true;

setup.aae.reg.gen = 1E0;
setup.aae.reg.dis = 1E0;
setup.aae.reg.wl2 = 1E-4;
setup.aae.reg.beta = 1E0;
setup.aae.reg.orth = 1E0;
setup.aae.reg.comp = 1E0;
setup.aae.reg.cls = 1E1;
setup.aae.reg.clust = 1E0;

setup.aae.valFreq = 50;
setup.aae.valSize = [2 5];
setup.aae.lrFreq = 250;
setup.aae.lrFactor = 0.5;

setup.aae.nDraw = 1;
setup.aae.zDim = nCodes;
setup.aae.xDim = length( setup.data.tFine );
setup.aae.cLabels = categorical( 0:length(classSizes) );
setup.aae.cDim = length( setup.aae.cLabels );
setup.aae.fda = setup.fda;

setup.aae.variational = true;
setup.aae.adversarial = true;
setup.aae.l2regularization = false;
setup.aae.orthogonal = true;
setup.aae.keyCompLoss = false;
setup.aae.clusterLoss = true;
setup.aae.useVarMean = true;
setup.aae.classifier = 'Network';

% encoder network parameters
setup.aae.enc.type = 'Convolutional'; %'Convolutional'; % 
setup.aae.enc.learnRate = 0.01;
setup.aae.enc.input = setup.aae.xDim;
setup.aae.enc.outZ = setup.aae.zDim*(setup.aae.variational + 1);
setup.aae.enc.projectionSize = setup.aae.xDim; % [ setup.aae.xDim sigDim 1 ];
setup.aae.enc.nHidden = 1;
setup.aae.enc.filterSize = 3;
setup.aae.enc.nFilters = 8;
setup.aae.enc.stride = 2;
setup.aae.enc.scale = 0.2;
setup.aae.enc.dropout = 0.1;
setup.aae.enc.maxPooling = false;
setup.aae.enc.poolSize = 3;
setup.aae.enc.nFC = 50;

% decoder network parameters
setup.aae.dec.type = 'Convolutional'; %'FullyConnected'; % 
setup.aae.dec.learnRate = 0.01;

setup.aae.dec.input = setup.aae.zDim;
setup.aae.dec.outX = setup.aae.xDim;
setup.aae.dec.projectionSize = 5; % [ 5 sigDim 1 ];
setup.aae.dec.nHidden = 1;
setup.aae.dec.filterSize = 3;
setup.aae.dec.nFilters = 8;
setup.aae.dec.stride = 2;
setup.aae.dec.scale = 0.2;
setup.aae.dec.dropout = 0;
setup.aae.dec.maxPooling = false;
setup.aae.dec.poolSize = 3;
setup.aae.dec.nFC = 50;

% discriminator network parameters
setup.aae.dis.learnRate = 0.01;
setup.aae.dis.dropout = 0.2;
setup.aae.dis.input = setup.aae.zDim;
setup.aae.dis.nHidden = 1;
setup.aae.dis.nFC = 5*setup.aae.zDim;

% classifier network parameters
setup.aae.cls.learnRate = 0.01;
setup.aae.cls.dropout = 0.0;
setup.aae.cls.input = setup.aae.zDim;
setup.aae.cls.output = setup.aae.cDim;
setup.aae.cls.nHidden = 1;
setup.aae.cls.nFC = 5*setup.aae.zDim;


end