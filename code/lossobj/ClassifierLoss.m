classdef ClassifierLoss < LossFunction
    % Subclass for classifier loss using an auxiliary network

    properties
        ZDim                % latent codes dimension size
        CDim                % number of possible classes
        NumHidden           % number of hidden layers
        NumFC               % number of fully connected nodes at widest
        FCFactor            % node ratio specifying the power 2 index
        Scale               % leaky Relu scale
        Dropout             % dropout rate
        InitLearningRate    % initial learning rate
        ModelType           % type of classifier model
        CLabels             % categorical labels
    end

    methods

        function self = ClassifierLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name            char {mustBeText}
                superArgs.?LossFunction
                args.modelType  char ...
                    {mustBeMember( args.modelType, ...
                                    {'Network', ...
                                     'Logistic', ...
                                     'Fisher', ...
                                     'SVM'} )} = 'Network'
                args.nHidden    double ...
                            {mustBeInteger, mustBePositive} = 1
                args.nFC        double ...
                            {mustBeInteger, mustBePositive} = 100
                args.fcFactor   double ...
                            {mustBeInteger, mustBePositive} = 1
                args.scale      double ...
                            {mustBeInRange(args.scale, 0, 1)} = 0.2
                args.dropout    double ...
                            {mustBeInRange(args.dropout, 0, 1)} = 0.1
                args.initLearningRate     double ...
                    {mustBeInRange(args.initLearningRate, 0, 1)} = 0.001
            end

            superArgsCell = namedargs2cell( superArgs );
            netAssignments = {'Encoder', 'Decoder', name};

            isNet = strcmp( args.modelType, 'Network' );

            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Auxiliary', ...
                                 input = 'Z-Y', ...
                                 lossNets = netAssignments, ...
                                 hasNetwork = isNet, ...
                                 hasState = isNet );

            self.NumHidden = args.nHidden;
            self.NumFC = args.nFC;
            self.FCFactor = args.fcFactor;
            self.Scale = args.scale;
            self.Dropout = args.dropout;
            self.ModelType = args.modelType;

            if isNet
                self.InitLearningRate = args.initLearningRate;
            else
                self.InitLearningRate = 0;
            end

        end


        function self = setDimensions( self, thisModel )
            % Store the Z dimension input once known
            arguments
                self            ClassifierLoss
                thisModel       AEModel
            end

            self.ZDim = thisModel.ZDim;
            self.CDim = thisModel.CDim;
            self.CLabels = categorical( 1:self.CDim );

        end


        function net = initNetwork( self )
            % Generate an initialized network
            arguments
                self
            end

            if ~strcmp( self.ModelType, 'Network' )
                eid = 'classifierLoss:NotNetwork';
                msg = 'This classifier function does not use a network.';
                throwAsCaller( MException(eid,msg) );
            end 

            % create the input layer
            layers = featureInputLayer( self.ZDim, 'Name', 'in' );
            
            % create the hidden layers
            for i = 1:self.NumHidden
                nNodes = fix( self.NumFC*2^(self.FCFactor*(1-i)) );
                layers = [ layers; ...
                    fullyConnectedLayer( nNodes, 'Name', ['fc' num2str(i)] )
                    batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                    leakyReluLayer( self.Scale, 'Name', ['relu' num2str(i)] )
                    dropoutLayer( self.Dropout, 'Name', ['drop' num2str(i)] )
                    ]; %#ok<AGROW> 
            end
            
            % create final layers
            layers = [ layers; ...    
                            fullyConnectedLayer( self.CDim, 'Name', 'fcout' )
                            softmaxLayer( 'Name', 'out' )
                            ];
            
            lgraph = layerGraph( layers );
            net = dlnetwork( lgraph );
            
        end


        function [ loss, state ] = calcLoss(  self, net, dlZGen, dlC )
            % Calculate the classifier loss
            arguments
                self     ClassifierLoss
                net      dlnetwork
                dlZGen   dlarray  % generated latent distribtion
                dlC      dlarray  % actual distribution
            end

            if self.HasNetwork                    
                [ loss, state ] = self.networkLoss( net, dlZGen, dlC );
            
            else
                loss = self.nonNetworkLoss( self.ModelType, dlZGen, dlC );
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
            switch self.ModelType
                case 'Logistic'
                    model = fitclinear( ZGen, C, Learner = "logistic" );
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
