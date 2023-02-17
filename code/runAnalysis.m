% Run the analysis

setup = initSetup;

% first investigation
name = 'test5';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "data.args.NormalizedPts", ...
               "model.args.UsesFdCoefficients", ...
               "model.args.lossFcns.reconrough.args.useLoss", ...
               "model.args.lossFcns.zorth.args.useLoss"];
values = {[11,13,15,17], ...
          {false, true}, ...
          {false, true}, ...
          {false, true}};

myInvestigation = Investigation( name, path, parameters, values, setup );

