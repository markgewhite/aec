% Run the analysis

setup = initSetup;

% first investigation
name = 'Test';
path = pwd;
%parameters = [ "model.args.ZDim", ...
%               "lossFcns.cls.args.useLoss" ];
%values = [ {2:6} {{false, true}} ];

parameters = [ "model.args.auxModel" ];
values = {["SVM", "Fisher"]};

myInvestigation = investigation( name, path, parameters, values, setup );