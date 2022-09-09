% Run the analysis

setup = initSetup;

% first investigation
name = 'cv_test';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.class", "model.args.IdenticalPartitions" ];
values = {{@FCModel},{true,false}};

myInvestigation = Investigation( name, path, parameters, values, setup );

