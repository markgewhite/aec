% Run the analysis

setup = initSetup;

% first investigation
name = 'JumpsGRF-005-ReliabilityTest(NoInnerProduct)';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

%parameters = "model.args.ZDim";
%values = {[1:6]};

parameters = [ "model.args.trainer.batchSize" ];
values = {{40}};

myInvestigation = Investigation( name, path, parameters, values, setup );

