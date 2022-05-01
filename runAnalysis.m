% Run the analysis

setup = initSetup;

% first investigation
name = 'Test';
path = pwd;
parameters = [ "model.args.ZDim", ...
               "trainer.args.nEpochs" ];
values = [ {2:6} {50:50:200} ];

myInvestigation = investigation( name, path, parameters, values, setup );