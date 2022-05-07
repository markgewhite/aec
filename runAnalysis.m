% Run the analysis

setup = initSetup;

% first investigation
name = 'Test';
path = fileparts( which('runAnalysis.m') );
path = [path '/results/'];

%parameters = [ "model.args.ZDim", ...
%               "lossFcns.cls.args.useLoss" ];
%values = [ {2:6} {{false, true}} ];

parameters = [ "model.class" ];
values = {{@lstmModel, @convModel, @tcnModel, @fcModel, @pcaModel}};

myInvestigation = investigation( name, path, parameters, values, setup );