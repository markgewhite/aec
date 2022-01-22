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
setup.aae.nEpochs = 500;
setup.aae.verbose = false;
setup.aae.nEpochsPretraining = setup.aae.nEpochs+1;
setup.randomSeed = 123;

% define optimizable variables
varDef(1) = optimizableVariable( 'enc__dropout', ...
        [0.0 0.9], ...
        'Type', 'real', 'Optimize', true );
varDef(2) = optimizableVariable( 'enc__filterSize', ...
        [2 7], ...
        'Type', 'integer', 'Optimize', true );
varDef(3) = optimizableVariable( 'enc__stride', ...
        [1 3], ...
        'Type', 'integer', 'Optimize', true );
varDef(4) = optimizableVariable( 'enc__nFilters', ...
        [1 32], ...
        'Type', 'integer', 'Optimize', true );
varDef(5) = optimizableVariable( 'enc__nHidden', ...
        [1 3], ...
        'Type', 'integer', 'Optimize', true );

% setup objective function
objFcn = @(x) objFcnAE( x, setup );

% run optimisation
output = bayesopt( objFcn, varDef, ...
            'MaxObjectiveEvaluations', 50 );




