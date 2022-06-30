% Run the analysis

setup = initSetupPC;

% first investigation
name = 'JumpsGRF-Test';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.class" ];
values = {{@FCModel}};

myInvestigation = Investigation( name, path, parameters, values, setup );

