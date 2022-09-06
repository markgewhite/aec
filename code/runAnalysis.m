% Run the analysis

setup = initSetup;

% first investigation
name = 'ucr3';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.class", "data.args.SetID" ];
values = {{@FullPCAModel}, 87:128};

myInvestigation = Investigation( name, path, parameters, values, setup );

