% Run the analysis

setup = initSetup;

% first investigation
name = 'test4';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "data.args.NormalizedPts", ...
               "model.args.UsesFdCoefficients" ];
%values = {[5, 10, 20, 50, 100]};
values = {[11], {true, false}};

myInvestigation = Investigation( name, path, parameters, values, setup );

