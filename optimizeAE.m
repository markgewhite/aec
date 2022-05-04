% ************************************************************************
% Script: optimizeAE
%
% ************************************************************************

clear;


% initalise setup
setup = initSetup;
setup.opt.objective = 'ReconLoss';
setup.model.args.ShowPlots = false;

% define optimizable variables
varDef(1) = optimizableVariable( 'model_args_ZDim', ...
        [1 15], ...
        'Type', 'integer', 'Optimize', true );


% setup objective function
objFcn = @(x) objectiveFcnAE( x, setup );

% run optimisation
output = bayesopt( objFcn, varDef, ...
            'NumCoupledConstraints', 1, ...
            'MaxObjectiveEvaluations', 30 );




