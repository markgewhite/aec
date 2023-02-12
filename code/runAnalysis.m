% Run the analysis

setup = initSetup;

% first investigation
name = 'test';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "data.args.NormalizedPts" ];
%values = {[5, 10, 20, 50, 100]};
values = {[11]};

myInvestigation = Investigation( name, path, parameters, values, setup );

