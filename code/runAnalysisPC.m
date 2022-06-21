% Run the analysis

setup = initSetupPC;

% first investigation
name = 'JumpsGRF(DemoB)';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "lossFcns.cls.args.useLoss" ];
values = {{true}};

myInvestigation = Investigation( name, path, parameters, values, setup );