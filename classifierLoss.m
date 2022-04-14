% ************************************************************************
% Class: classifierLoss
%
% Subclass for classifier loss using an auxiliary network
%
% ************************************************************************

classdef classifierLoss < lossFunction

    properties
        ZDim                % latent codes dimension size
        CDim                % number of possible classes
        nHidden             % number of hidden layers
        nFC                 % number of fully connected nodes at widest
        fcFactor            % node ratio specifying the power 2 index
        scale               % leaky Relu scale
        dropout             % dropout rate
        initLearningRate    % initial learning rate
        modelType           % type of classifier model
        CLabels             % categorical labels
    end

    methods

        function self = classifierLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name            char {mustBeText}
                superArgs.?lossFunction
                args.modelType  char ...
                    {mustBeMember( args.modelType, ...
                                    {'Network', ...
                                     'Fisher', ...
                                     'SVM'} )} = 'Network'
                args.ZDim       double ...
                            {mustBeInteger, mustBePositive} = 4
                args.CDim       double ...
                            {mustBeInteger, mustBePositive} = 2
                args.nHidden    double ...
                            {mustBeInteger, mustBePositive} = 2
                args.nFC        double ...
                            {mustBeInteger, mustBePositive} = 128
                args.fcFactor   double ...
                            {mustBeInteger, mustBePositive} = 2
                args.scale      double ...
                            {mustBeInRange(args.scale, 0, 1)} = 0.2
                args.dropout    double ...
                            {mustBeInRange(args.dropout, 0, 1)} = 0.1
                args.initLearningRate     double ...
                    {mustBeInRange(args.initLearningRate, 0, 1)} = 0.001
            end

            superArgsCell = namedargs2cell( superArgs );
            netAssignments = {'encoder', name};

            self = self@lossFunction( name, superArgsCell{:}, ...
                                 type = 'Auxiliary', ...
                                 input = 'Z-Y', ...
                                 lossNets = netAssignments, ...
                                 hasNetwork = true, ...
                                 hasState = true );

            self.ZDim = args.ZDim;
            self.CDim = args.CDim;
            self.nHidden = args.nHidden;
            self.nFC = args.nFC;
            self.fcFactor = args.fcFactor;
            self.scale = args.scale;
            self.dropout = args.dropout;
            self.modelType = args.modelType;

            switch args.modelType
                case 'Network'
                    self.initLearningRate = args.initLearningRate;

                otherwise
                    % not a network, will use a fixed model
                    self.initLearningRate = 0;

            end

            self.CLabels = categorical( 1:self.CDim );

        end


        function net = initNetwork( self )
            % Generate an initialized network
            arguments
                self
            end

            if ~strcmp( self.modelType, 'Network' )
                eid = 'classifierLoss:NotNetwork';
                msg = 'This classifier function does not use a network.';
                throwAsCaller( MException(eid,msg) );
            end 

            % create the input layer
            layers = featureInputLayer( self.ZDim, 'Name', 'in' );
            
            % create the hidden layers
            for i = 1:self.nHidden
                nNodes = fix( self.nFC*2^(self.fcFactor*(1-i)) );
                layers = [ layers; ...
                    fullyConnectedLayer( nNodes, 'Name', ['fc' num2str(i)] )
                    batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                    leakyReluLayer( self.scale, 'Name', ['relu' num2str(i)] )
                    dropoutLayer( self.dropout, 'Name', ['drop' num2str(i)] )
                    ]; %#ok<AGROW> 
            end
            
            % create final layers
            layers = [ layers; ...    
                            fullyConnectedLayer( self.CDim, 'Name', 'fcout' )
                            sigmoidLayer( 'Name', 'out' )
                            ];
            
            lgraph = layerGraph( layers );
            net = dlnetwork( lgraph );
            
        end


        function [ self, loss, state ] = calcLoss(  self, net, dlZGen, dlC )
            % Calculate the classifier loss
            arguments
                self     classifierLoss
                net      dlnetwork
                dlZGen   dlarray  % generated latent distribtion
                dlC      dlarray  % actual distribution
            end

            if self.hasNetwork                    
                [ loss, state ] = self.networkLoss( net, dlZGen, dlC );
            
            else
                loss = self.nonNetworkLoss( self.modelType, dlZGen, dlC );
                state = [];

            end

    
        end


    end

    methods (Access = protected)

        function [ loss, state ] = networkLoss( self, net, dlZGen, dlC )

            % get the network's predicted class
            [ dlCGen, state ] = forward( net, dlZGen );

            % hotcode the actual class 
            dlCActual = dlarray( ...
                onehotencode( self.CLabels(dlC), 1 ), 'CB' );
            
            % compute the cross entropy (classifier) loss
            loss = crossentropy( dlCGen, dlCActual, ...
                                  'TargetCategories', 'Independent' );

        end


        function loss = nonNetworkLoss( self, dlZGen, dlC )

            % convert to double for models which don't take dlarrays
            ZGen = double(extractdata( dlZGen ))';
            C = double(extractdata( dlC ));
            
            % fit the appropriate model
            switch self.modelType
                case 'Fisher'
                    model = fitcdiscr( ZGen, C );
                case 'SVM'
                    model = fitcecoc( ZGen, C );
            end
            
            % compute the training loss
            loss = resubLoss( model );
        
        end

    end

end
