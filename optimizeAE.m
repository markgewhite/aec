% ************************************************************************
% Script: optimizeAE
%
% ************************************************************************

clear;


N = 200;
classSizes = [ N N N ];
nDim = 1;
nCodes = 4;

% initalise setup
setup = initializeAE( nCodes, classSizes, nDim );
setup.aae.nEpochs = 100;
setup.aae.verbose = false;
setup.aae.nEpochsPretraining = setup.aae.nEpochs+1;
setup.randomSeed = 123;

% define optimizable variables
varDef(1) = optimizableVariable( 'enc_x_dropout', ...
        [0.0 0.9], ...
        'Type', 'real', 'Optimize', false );
varDef(2) = optimizableVariable( 'enc_x_filterSize', ...
        [2 20], ...
        'Type', 'integer', 'Optimize', true );
varDef(3) = optimizableVariable( 'enc_x_stride', ...
        [1 3], ...
        'Type', 'integer', 'Optimize', false );
varDef(4) = optimizableVariable( 'enc_x_nFilters', ...
        [1 64], ...
        'Type', 'integer', 'Optimize', true );
varDef(5) = optimizableVariable( 'enc_x_nHidden', ...
        [1 3], ...
        'Type', 'integer', 'Optimize', false );
varDef(6) = optimizableVariable( 'enc_x_scale', ...
        [0.0 1.0], ...
        'Type', 'real', 'Optimize', false );
varDef(7) = optimizableVariable( 'nEpochs', ...
        [1 500], ...
        'Type', 'integer', 'Optimize', false );

% setup objective function
objFcn = @(x) objFcnAE( x, setup );

% run optimisation
output = bayesopt( objFcn, varDef, ...
            'MaxObjectiveEvaluations', 30 );




