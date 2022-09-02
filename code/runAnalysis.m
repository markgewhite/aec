% Run the analysis

setup = initSetup;

% first investigation
name = 'new';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.class" ];
values = {{@FCModel, @FullPCAModel, @ConvolutionalModel}};

myInvestigation = Investigation( name, path, parameters, values, setup );

