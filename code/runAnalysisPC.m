% Run the analysis

setup = initSetupPC;

% first investigation
name = 'JumpsGRF-003-ReliabilityTestWithZOrth';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.class" ];
values = {{@FCModel}};

myInvestigation = Investigation( name, path, parameters, values, setup );

