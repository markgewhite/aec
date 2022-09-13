% Run the analysis

setup = initSetup;

% first investigation
name = 'test_kfoldrepeats';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.class" ];
values = {{@PCAModel,@FCModel,@ConvolutionalModel}};

myInvestigation = Investigation( name, path, parameters, values, setup );

