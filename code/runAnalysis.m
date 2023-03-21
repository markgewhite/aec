% Run the analysis

setup = initSetup;

% first investigation
name = 'test';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.args.ZDim" ];
values = {[3 5]};


myInvestigation = Investigation( name, path, parameters, values, setup );

myInvestigation.run;



