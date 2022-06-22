% Run the analysis

setup = initSetup;

% first investigation
name = 'JumpsGRF(ReliabilityTest5PCA)';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

%parameters = [ "model.args.ZDim", ...
%               "lossFcns.cls.args.useLoss" ];
%values = [ {2:6} {{false, true}} ];

parameters = [ "lossFcns.cls.args.useLoss" ];
values = {{false,true}};

myInvestigation = Investigation( name, path, parameters, values, setup );