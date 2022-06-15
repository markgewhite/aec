% Run the analysis

setup = initSetup;

% first investigation
name = 'JumpsGRF(Test)';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

%parameters = [ "model.args.ZDim", ...
%               "lossFcns.cls.args.useLoss" ];
%values = [ {2:6} {{false, true}} ];

parameters = [ "model.class" ];
values = {{@FCModel}};

myInvestigation = Investigation( name, path, parameters, values, setup );