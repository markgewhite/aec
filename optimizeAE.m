% ************************************************************************
% Script: optimizeAE
%
% ************************************************************************

clear;


% initalise setup
setup = initSetup;
setup.opt.objective = 'ReconLossSmoothed';
setup.model.args.ShowPlots = false;
setup.trainer.args.nEpochs = 100;

% define optimizable variables
varDef(1) = optimizableVariable( 'model_args_ZDim', ...
        [1 15], ...
        'Type', 'integer', 'Optimize', false );

varDef(2) = optimizableVariable( 'lossFcns_smooth_args_window', ...
        [3 41], ...
        'Type', 'integer', 'Optimize', false );

varDef(4) = optimizableVariable( 'lossFcns_smooth_args_Lambda', ...
        [1E-5 1E5], ...
        'Type', 'real', 'Transform', 'log', 'Optimize', true );


% setup objective function
objFcn = @(x) objectiveFcnAE( x, setup );

% run optimisation
output = bayesopt( objFcn, varDef, ...
            'NumCoupledConstraints', 1, ...
            'MaxObjectiveEvaluations', 30 );




