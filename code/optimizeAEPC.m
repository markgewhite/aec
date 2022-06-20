% ************************************************************************
% Script: optimizeAE
%
% ************************************************************************

clear;


% initalise setup
setup = initSetupPC;
setup.opt.objective = 'ReconLossRegular';
setup.model.args.ShowPlots = false;
setup.model.args.trainer.numEpochs = 100;
setup.model.args.trainer.updateFreq = 1E4;
setup.model.class = @FCModel;

% define optimizable variables
varDef(1) = optimizableVariable( 'model_args_ZDim', ...
        [1 30], ...
        'Type', 'integer', 'Optimize', true );


% setup objective function
objFcn = @(x) objectiveFcnAE( x, setup );

% run optimisation
output = bayesopt( objFcn, varDef, ...
            'NumCoupledConstraints', 1, ...
            'MaxObjectiveEvaluations', 30 );




