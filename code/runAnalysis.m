% Run the analysis

setup = initSetup;
setup.data.args.NormalizedPts = 11;

% first investigation
name = 'test2';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "data.args.NormalizedPts", ...
               "model.args.HasBranchedDecoder"];
values = {[21], ...
          [false true]};
%parameters = [ "data.args.NormalizedPts"];
%values = {[21 51]};


myInvestigation = Investigation( name, path, parameters, values, setup );

