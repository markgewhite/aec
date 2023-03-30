% Run the analysis

setup = initSetup;

% first investigation
name = 'test2';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.args.ZDim", ...
               ];
values = {[2 5], ...
          };


myInvestigation = Investigation( name, path, parameters, values, setup );

myInvestigation.run;



