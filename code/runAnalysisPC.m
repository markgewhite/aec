% Run the analysis

setup = initSetupPC;

% first investigation
name = 'test_destroyed3';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

%parameters = "model.args.ZDim";
%values = {[1:6]};

parameters = [ "model.args.trainer.NumEpochsPreTrn" ];
values = {{0}};

myInvestigation = Investigation( name, path, parameters, values, setup );

