classdef RegressionLoss < LossFunction
    % Subclass for a regression loss using an auxiliary network

    properties
        ZDimAux             % latent codes dimension size
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

        function self = RegressionLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name            char {mustBeText}
                superArgs.?LossFunction
                args.ModelType  char ...
                    {mustBeMember( args.ModelType, ...
                                    {'Network', ...
                                     'LR'} )} = 'Network'
                args.NumHidden    double ...
                            {mustBeInteger, mustBePositive} = 2
                args.NumFC        double ...
                            {mustBeInteger, mustBePositive} = 16
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
            netAssignments = {'Encoder', name};

            isNet = strcmp( args.ModelType, 'Network' );

            self = self@LossFunction( name, superArgsCell{:}, ...
                                 Type = 'Auxiliary', ...
                                 Input = {'dlZAux', 'dlY'}, ...
                                 LossNets = netAssignments, ...
                                 HasNetwork = isNet, ...
                                 HasState = isNet, ...
                                 YLim = [0 0.2] );

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
                self            RegressionLoss
                thisModel       AEModel
            end

            self.ZDimAux = thisModel.ZDimAux;

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
                if nNodes == 0
                    eid = 'RegressiomLoss:NetworkDesign';
                    msg = 'Hidden layer has zero nodes.';
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
                       fullyConnectedLayer( 1, 'Name', 'fcout' )
                       ];
            
            lgraph = layerGraph( layers );
            net = dlnetwork( lgraph );
            
        end


        function [ loss, state ] = calcLoss(  self, net, dlZGen, dlY )
            % Calculate the classifier loss
            arguments
                self     RegressionLoss
                net      dlnetwork
                dlZGen   dlarray  % generated latent distribtion
                dlY      dlarray  % actual distribution
            end

            dlZAux = dlZGen( 1:self.ZDimAux, : );

            if self.HasNetwork                    
                [ loss, state ] = self.networkLoss( net, dlZAux, dlY );
            
            else
                loss = self.nonNetworkLoss( self.ModelType, dlZAux, dlY );
                state = [];

            end

    
        end


    end

    methods (Access = protected)

        function [ loss, state ] = networkLoss( self, net, dlZ, dlY )

            % get the network's predicted class
            [ dlYHat, state ] = forward( net, dlZ );
           
            % compute the MSE loss
            loss = mean((dlYHat - dlY).^2);

        end


        function loss = nonNetworkLoss( self, dlZ, dlY )

            % convert to double for models which don't take dlarrays
            Z = double(extractdata( dlZ ))';
            Y = double(extractdata( dlY ));
            
            % fit the appropriate model
            switch self.ModelType
                case 'LR'
                    model = fitlm( Z, Y );
            end
            
            % compute the training loss
            loss = resubLoss( model );
        
        end

    end

end
