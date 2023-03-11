% Run the analysis

setup = initSetup;
setup.data.args.NormalizedPts = 11;

% first investigation
name = 'decoderBranching';
path = fileparts( which('code/runAnalysis.m') );
path = [path '/../results/'];

parameters = [ "model.args.HasBranchedDecoder", ...
               "model.args.HasDecoderMasking"];
values = {{false, true}, ...
          {false, true}};

%parameters = [ "model.class"];
%values = {{@MultiNetFCModel}};

myInvestigation = Investigation( name, path, parameters, values, setup );

