% Run the analysis

setup = initSetup;

% first investigation
name = 'TestAdaptive';
path = fileparts( which('runAnalysis.m') );
path = [path '/results/'];

%parameters = [ "model.args.ZDim", ...
%               "lossFcns.cls.args.useLoss" ];
%values = [ {2:6} {{false, true}} ];

parameters = [ "model.class" ];
%values = {{@fcModel, @lstmfcModel, @lstmModel, @convModel, @tcnModel, @pcaModel}};
values = {{@pcaModel}};

myInvestigation = investigation( name, path, parameters, values, setup );