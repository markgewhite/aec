% Run the analysis

setup = initSetup;
setup.data.args.NormalizedPts = 11;

% first investigation
name = 'test';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.args.NumHiddenDecoder", ...
               "model.args.NumFCDecoder", ...
               "model.args.ZDim" ];
values = {[1, 3], ...
          [128, 384], ...
          [3, 5, 10]};

myInvestigation = Investigation( name, path, parameters, values, setup );

