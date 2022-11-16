% Run the analysis

setup = initSetup;

% first investigation
name = 'test';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.class", "model.args.trainer.useParallelProcessing" ];
values = {{@FCModel}, {true }};

myInvestigation = Investigation( name, path, parameters, values, setup );

