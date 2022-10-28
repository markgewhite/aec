% Run the analysis

setup = initSetup;

% first investigation
name = 'test';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.class", ...
               "model.args.lossFcns.adv.args.distribution" ];
values = {{@FCModel}, {'Gaussian'}};

myInvestigation = Investigation( name, path, parameters, values, setup );

