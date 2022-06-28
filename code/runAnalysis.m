% Run the analysis

setup = initSetup;

% first investigation
name = 'JumpsGRF-Test';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

%parameters = [ "model.args.ZDim", ...
%               "lossFcns.cls.args.useLoss" ];
%values = [ {2:6} {{false, true}} ];

parameters = [ "model.class" ];
values = {{@FCModel}};

%parameters = [ "model.args.IsVAE", ...
%               "lossFcns.kl.args.useLoss", ...
%               "lossFcns.adv.args.useLoss" ];
%values = { {false,true}, ...
%           {false,true}, ...
%           {false,true} };

myInvestigation = Investigation( name, path, parameters, values, setup );

