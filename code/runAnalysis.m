% Run the analysis

setup = initSetup;

% first investigation
name = 'test_destroyed3';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

%parameters = "model.args.ZDim";
%values = {[1:6]};

parameters = [ "model.class" ];
values = {{@ConvolutionalModel, @FCModel}};

myInvestigation = Investigation( name, path, parameters, values, setup );

