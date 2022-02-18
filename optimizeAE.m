% ************************************************************************
% Script: optimizeAE
%
% ************************************************************************

clear;


% initalise setup
setup.opt.nCodes = 4;
setup.opt.dataSource = 'JumpVGRF';
setup.opt.randomSeed = 123;
setup.opt.objective = 'ClassificationError';
setup.opt.nEpochs = 500;

% define optimizable variables
varDef(1) = optimizableVariable( 'nEpochs', ...
        [1 500], ...
        'Type', 'integer', 'Optimize', false );


varDef(2) = optimizableVariable( 'enc_x_filterSize', ...
        [2 20], ...
        'Type', 'integer', 'Optimize', false );
varDef(3) = optimizableVariable( 'enc_x_stride', ...
        [1 3], ...
        'Type', 'integer', 'Optimize', false );
varDef(4) = optimizableVariable( 'enc_x_nFilters', ...
        [10 256], ...
        'Type', 'integer', 'Optimize', false );
varDef(5) = optimizableVariable( 'enc_x_nHidden', ...
        [0 2], ...
        'Type', 'integer', 'Optimize', false );
varDef(6) = optimizableVariable( 'enc_x_scale', ...
        [0.0 1.0], ...
        'Type', 'real', 'Optimize', false );
varDef(7) = optimizableVariable( 'enc_x_dropout', ...
        [0.0 0.5], ...
        'Type', 'real', 'Optimize', false );

varDef(8) = optimizableVariable( 'dis_x_nHidden', ...
        [0 5], ...
        'Type', 'integer', 'Optimize', false );
varDef(9) = optimizableVariable( 'dis_x_nFC', ...
        [10 100], ...
        'Type', 'integer', 'Optimize', false );
varDef(10) = optimizableVariable( 'dis_x_scale', ...
        [0 1.0], ...
        'Type', 'real', 'Optimize', false );
varDef(11) = optimizableVariable( 'dis_x_dropout', ...
        [0 0.9], ...
        'Type', 'real', 'Optimize', false );


varDef(12) = optimizableVariable( 'cls_x_nHidden', ...
        [0 5], ...
        'Type', 'integer', 'Optimize', false );
varDef(13) = optimizableVariable( 'cls_x_nFC', ...
        [5 300], ...
        'Type', 'integer', 'Optimize', false );
varDef(14) = optimizableVariable( 'cls_x_scale', ...
        [0 1.0], ...
        'Type', 'real', 'Optimize', true );
varDef(15) = optimizableVariable( 'cls_x_dropout', ...
        [0 0.9], ...
        'Type', 'real', 'Optimize', true );

varDef(16) = optimizableVariable( 'variational', ...
        ["false" "true"], ...
        'Type', 'categorical', 'Optimize', false );
varDef(17) = optimizableVariable( 'adversarial', ...
        ["false" "true"], ...
        'Type', 'categorical', 'Optimize', false );

varDef(18) = optimizableVariable( 'nKernels', ...
        [10 10000], ...
        'Type', 'integer', 'Transform', 'log', 'Optimize', false );
varDef(19) = optimizableVariable( 'candidateStart', ...
        [1 5], ...
        'Type', 'integer', 'Optimize', false );
varDef(20) = optimizableVariable( 'nCandidates', ...
        [1 9], ...
        'Type', 'integer', 'Optimize', false );
varDef(21) = optimizableVariable( 'enc_x_nFC', ...
        [10 1000], ...
        'Type', 'integer', 'Transform', 'log', 'Optimize', false );

% setup objective function
objFcn = @(x) objFcnAE( x, setup );

% run optimisation
output = bayesopt( objFcn, varDef, ...
            'NumCoupledConstraints', 1, ...
            'MaxObjectiveEvaluations', 30 );




