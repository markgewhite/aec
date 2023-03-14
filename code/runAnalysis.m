% Run the analysis

setup = initSetup;

% first investigation
name = 'test_Fukuchi';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.args.ZDim", ...
               "model.args.NumHiddenDecoder" ];
values = {[3 5 10], ...
          [1 2 5]};


myInvestigation = Investigation( name, path, parameters, values, setup );

