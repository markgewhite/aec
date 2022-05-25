% Run the analysis

setup = initSetup;

% first investigation
name = 'JumpsGRF(Adaptive-Classification)';
path = fileparts( which('runAnalysis.m') );
path = [path '/results/'];

%parameters = [ "model.args.ZDim", ...
%               "lossFcns.cls.args.useLoss" ];
%values = [ {2:6} {{false, true}} ];

parameters = [ "model.class" ];
values = {{@pcaModel, @fcModel, @convModel}};

myInvestigation = investigation( name, path, parameters, values, setup );