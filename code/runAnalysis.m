% Run the analysis

setup = initSetup;

% first investigation
name = 'test';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "data.args.NormalizedPts" ];
values = {[5, 10, 20]};

myInvestigation = Investigation( name, path, parameters, values, setup );

