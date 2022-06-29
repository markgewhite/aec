% Run the analysis

setup = initSetup;

% first investigation
name = 'JumpsGRF-Test';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = "model.args.ZDim";
values = {[1:6]};

%parameters = [ "model.class" ];
%values = {{@FullPCAModel,@FCModel}};

%parameters = [ "model.args.IsVAE", ...
%               "lossFcns.kl.args.useLoss", ...
%               "lossFcns.adv.args.useLoss" ];
%values = { {false,true}, ...
%           {false,true}, ...
%           {false,true} };

myInvestigation = Investigation( name, path, parameters, values, setup );

