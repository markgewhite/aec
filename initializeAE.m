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
setup.nEpochs = 1000;
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
setup.reg.cls = 1E0;
setup.reg.clust = 1E0;

setup.valFreq = 5;
setup.updateFreq = 25;
setup.lrFreq = 250;
setup.lrFactor = 0.5;
setup.valPatience = 50; % 50

setup.xDim = config.xDim;
setup.zDim = config.zDim;
setup.cDim = config.cDim;
setup.cLabels = config.cLabels;
setup.nChannels = config.nChannels;
setup.nDraw = config.nDraw;

setup.postTraining = true; % preTraining is set during training
setup.variational = false;
setup.adversarial = true;
setup.unimodal = false;
setup.wasserstein = false;
setup.l2regularization = false;
setup.orthogonal = true;
setup.keyCompLoss = false;
setup.clusterLoss = true;
setup.useVarMean = true;
setup.classifier = 'Network';
setup.validationFcn = 'Fisher';

setup.mmd.scale = 2;
setup.mmd.kernel = 'IMQ';
setup.mmd.baseType = 'Normal';

setup.fda = config.fda;



% encoder network parameters
setup.enc.type = 'TCN'; %'Convolutional'; % 
setup.enc.learnRate = 0.01;
if config.embedding
    setup.enc.input = config.nFeatures;
else
    setup.enc.input = config.xDim*config.nChannels;
    setup.enc.nChannels = config.nChannels;
end

setup.enc.outZ = config.zDim*(setup.variational + 1);
setup.enc.projectionSize = config.xDim; % [ config.xDim config.nChannels 1 ];
switch config.source
    case 'Synthetic'
        setup.enc.nHidden = 1;
        setup.enc.filterSize = 3;
        setup.enc.nFilters = 18;
        setup.enc.stride = 3;
        setup.enc.scale = 0.2;
        setup.enc.dropout = 0.1;

    case {'JumpVGRF', 'MSFT'}
        switch setup.enc.type
            case 'FullyConnected'
                setup.enc.nHidden = 3;
                setup.enc.nFC = 512;
                setup.enc.fcFactor = 2;
                setup.enc.scale = 0;
                setup.enc.dropout = 0.10;
            case 'Convolutional'
                setup.enc.nHidden = 1;
                setup.enc.filterSize = 3;
                setup.enc.nFilters = 220;
                setup.enc.stride = 2;
                setup.enc.scale = 0.4;
                setup.enc.dropout = 0.1;
            case 'TCN'
                setup.enc.projectionSize = [config.xDim setup.enc.nChannels]; 
                setup.enc.nHidden = 3;
                setup.enc.filterSize = 5;
                setup.enc.nFilters = 16;
                setup.enc.scale = 0.2;
                setup.enc.initialDropout = 0.1;
                setup.enc.dropout = 0.05;
        end
    otherwise
        error('Unrecognised data source');
end


% decoder network parameters
setup.dec.type = 'TCN'; %'FullyConnected'; % 
setup.dec.learnRate = 0.01;
setup.dec.input = config.zDim;
setup.dec.outX = [ config.xDim config.nChannels ];
setup.dec.projectionSize = 5; % [ 5 sigDim 1 ];
setup.dec.nFC = 50;
switch config.source
    case 'Synthetic'
        setup.dec.nHidden = 1;
        setup.dec.filterSize = 3;
        setup.dec.nFilters = 18;
        setup.dec.stride = 3;
        setup.dec.scale = 0.2;
        setup.dec.dropout = 0;

    case {'JumpVGRF', 'MSFT'}
        switch setup.dec.type
            case 'FullyConnected'
                setup.dec.nHidden = 2; % 1
                setup.dec.nFC = 64; % 32
                setup.dec.fcFactor = 2; % 1
                setup.dec.scale = 0.2;
                setup.dec.dropout = 0;
            case 'Convolutional'
                setup.dec.nHidden = 1;
                setup.dec.filterSize = 18;
                setup.dec.nFilters = 32;
                setup.dec.stride = 3;
                setup.dec.scale = 0.2;
                setup.dec.dropout = 0;
            case 'TCN'
                setup.dec.projectionSize = [ config.xDim 1 config.nChannels ];
                setup.dec.nHidden = 3;
                setup.dec.filterSize = 5;
                setup.dec.nFilters = 16;
                setup.dec.scale = 0.2;
                setup.dec.dropout = 0.0;

        end
    otherwise
        error('Unrecognised data source');
end


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
switch config.source
    case 'Synthetic'
        setup.cls.nHidden = 1;
        setup.cls.nFC = 100;
        setup.cls.scale = 0.2;
        setup.cls.dropout = 0.15;
    case {'JumpVGRF', 'MSFT'}
        setup.cls.nHidden = 1;
        setup.cls.nFC = 100;
        setup.cls.scale = 0.2;
        setup.cls.dropout = 0.2;
    otherwise
        error('Unrecognised data source');
end


end