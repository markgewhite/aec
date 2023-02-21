% Run the analysis

setup = initSetup;
setup.data.args.NormalizedPts = 11;

% first investigation
name = 'test';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.args.UsesFdCoefficients", ...
               "model.args.lossFcns.reconrough.args.useLoss", ...
               "model.args.lossFcns.zorth.args.useLoss"];
values = {{false, true}, ...
          {false, true}, ...
          {false, true}};

myInvestigation = Investigation( name, path, parameters, values, setup );

