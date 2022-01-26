% ************************************************************************
% Function: initializeAE
%
% Initialise the autoencoder setup
%
% Parameters:
%           config : data setup structure
%           
% Outputs:
%           setup : initialised setup structure
%
% ************************************************************************


function setup = initializeAE( config )

% AAE training parameters
setup.designFcn = @aaeDesign;
setup.gradFcn = @modelGradients;
setup.optimizer = 'ADAM';
setup.nEpochs = 1500;
setup.nEpochsPretraining = 10;
setup.batchSize = 100;
setup.beta1 = 0.9;
setup.beta2 = 0.999;
setup.verbose = true;

setup.reg.gen = 1E0;
setup.reg.dis = 1E0;
setup.reg.wl2 = 1E-4;
setup.reg.beta = 1E0;
setup.reg.orth = 1E0;
setup.reg.comp = 1E0;
setup.reg.cls = 1E1;
setup.reg.clust = 1E0;

setup.valFreq = 25;
setup.valSize = [2 5];
setup.lrFreq = 250;
setup.lrFactor = 0.5;

setup.xDim = config.xDim;
setup.zDim = config.zDim;
setup.cDim = config.cDim;
setup.cLabels = config.cLabels;
setup.nDraw = config.nDraw;

setup.variational = true;
setup.adversarial = true;
setup.l2regularization = false;
setup.orthogonal = true;
setup.keyCompLoss = false;
setup.clusterLoss = true;
setup.useVarMean = true;
setup.classifier = 'Network';

setup.fda = config.fda;

% encoder network parameters
setup.enc.type = 'Convolutional'; %'Convolutional'; % 
setup.enc.learnRate = 0.01;
setup.enc.input = config.xDim;
setup.enc.outZ = config.zDim*(setup.variational + 1);
setup.enc.projectionSize = config.xDim; % [ setup.xDim sigDim 1 ];
setup.enc.nHidden = 1;
setup.enc.filterSize = 3;
setup.enc.nFilters = 18;
setup.enc.stride = 3;
setup.enc.scale = 0.2;
setup.enc.dropout = 0.1;
setup.enc.nFC = 50;

% decoder network parameters
setup.dec.type = 'Convolutional'; %'FullyConnected'; % 
setup.dec.learnRate = 0.01;

setup.dec.input = config.zDim;
setup.dec.outX = config.xDim;
setup.dec.projectionSize = 5; % [ 5 sigDim 1 ];
setup.dec.nHidden = 1;
setup.dec.filterSize = 18;
setup.dec.nFilters = 32;
setup.dec.stride = 3;
setup.dec.scale = 0.2;
setup.dec.dropout = 0;
setup.dec.nFC = 50;

% discriminator network parameters
setup.dis.learnRate = 0.01;
setup.dis.input = config.zDim;
setup.dis.nHidden = 4;
setup.dis.nFC = 100;
setup.dis.scale = 0.3;
setup.dis.dropout = 0.15;

% classifier network parameters
setup.cls.learnRate = 0.01;
setup.cls.input = config.zDim;
setup.cls.output = config.cDim;
setup.cls.nHidden = 1;
setup.cls.nFC = 300;
setup.cls.scale = 0.3;
setup.cls.dropout = 0.15;


end