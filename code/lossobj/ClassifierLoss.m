classdef ClassifierLoss < LossFunction
    % Subclass for classifier loss using an auxiliary network

    properties
        ZDimAux             % latent codes dimension size
        CDim                % number of possible classes
        NumHidden           % number of hidden layers
        NumFC               % number of fully connected nodes at widest
        FCFactor            % node ratio specifying the power 2 index
        ReluScale           % leaky Relu scale
        Dropout             % dropout rate
        HasBatchNormalization % if normalizing batches
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
                args.ModelType  char ...
                    {mustBeMember( args.ModelType, ...
                                    {'Network', ...
                                     'Logistic', ...
                                     'Fisher', ...
                                     'SVM'} )} = 'Network'
                args.NumHidden    double ...
                            {mustBeInteger, mustBePositive} = 4
                args.NumFC        double ...
                            {mustBeInteger, mustBePositive} = 64
                args.FCFactor     double ...
                            {mustBeInteger, mustBePositive} = 1
                args.ReluScale    double ...
                            {mustBeInRange(args.ReluScale, 0, 1)} = 0.2
                args.Dropout    double ...
                            {mustBeInRange(args.Dropout, 0, 0.9)} = 0.1
                args.HasBatchNormalization  logical = true
                args.InitLearningRate     double ...
                    {mustBeInRange(args.InitLearningRate, 0, 1)} = 0.001
            end

            superArgsCell = namedargs2cell( superArgs );
            netAssignments = {'Encoder', 'Decoder', name};

            isNet = strcmp( args.ModelType, 'Network' );

            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Auxiliary', ...
                                 input = 'Z-Y', ...
                                 lossNets = netAssignments, ...
                                 hasNetwork = isNet, ...
                                 hasState = isNet );

            self.NumHidden = args.NumHidden;
            self.NumFC = args.NumFC;
            self.FCFactor = args.FCFactor;
            self.ReluScale = args.ReluScale;
            self.Dropout = args.Dropout;
            self.HasBatchNormalization = args.HasBatchNormalization;
            self.ModelType = args.ModelType;

            if isNet
                self.InitLearningRate = args.InitLearningRate;
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

            self.ZDimAux = thisModel.ZDimAux;
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
            layers = featureInputLayer( self.ZDimAux, 'Name', 'in' );
            
            % create the hidden layers
            for i = 1:self.NumHidden

                nNodes = fix( self.NumFC*2^(self.FCFactor*(1-i)) );
                if nNodes <= self.CDim
                    eid = 'ClassifierLoss:NetworkDesign';
                    msg = 'Hidden layer has fewer nodes than classes.';
                    throwAsCaller( MException(eid,msg) );
                end

                if self.HasBatchNormalization
                    layers = [ layers; ...
                        fullyConnectedLayer( nNodes, 'Name', ['fc' num2str(i)] )
                        batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                        leakyReluLayer( self.ReluScale, 'Name', ['relu' num2str(i)] )
                        dropoutLayer( self.Dropout, 'Name', ['drop' num2str(i)] )
                        ]; %#ok<AGROW> 
                else
                    layers = [ layers; ...
                        fullyConnectedLayer( nNodes, 'Name', ['fc' num2str(i)] )
                        leakyReluLayer( self.ReluScale, 'Name', ['relu' num2str(i)] )
                        dropoutLayer( self.Dropout, 'Name', ['drop' num2str(i)] )
                        ]; %#ok<AGROW>
                end

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

            dlZAux = dlZGen( 1:self.ZDimAux, : );

            if self.HasNetwork                    
                [ loss, state ] = self.networkLoss( net, dlZAux, dlC );
            
            else
                loss = self.nonNetworkLoss( self.ModelType, dlZAux, dlC );
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
