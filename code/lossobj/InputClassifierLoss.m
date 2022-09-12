classdef InputClassifierLoss < LossFunction
    % Subclass for classifier loss with X as the input for comparison purposes

    properties
        XDim                % input dimension size
        XChannels           % input channels
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

        function self = InputClassifierLoss( name, superArgs, args )
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
                            {mustBeInRange(args.dropout, 0, 1)} = 0.2
                args.initLearningRate     double ...
                    {mustBeInRange(args.initLearningRate, 0, 1)} = 0.01
            end

            superArgsCell = namedargs2cell( superArgs );
            netAssignments = {name};

            isNet = strcmp( args.modelType, 'Network' );

            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Comparator', ...
                                 input = 'X-Y', ...
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
                self            InputClassifierLoss
                thisModel       AEModel
            end

            self.XDim = thisModel.XInputDim;
            self.XChannels = thisModel.XChannels;
            self.CDim = thisModel.CDim;
            self.CLabels = categorical( 1:self.CDim );

        end
        
        
        function net = initNetwork( self, encoder )
            % Generate an initialized network
            arguments
                self        InputClassifierLoss
                encoder     dlnetwork
            end

            lgraph = layerGraph( encoder );
            lastName = lgraph.Layers(end).Name;
           
            % define replacement layers
            newLayers = [ fullyConnectedLayer( self.CDim, 'Name', 'fcout' )
                          softmaxLayer( 'Name', 'out' ) ];

            lgraph = replaceLayer( lgraph, lastName, newLayers );
            net = dlnetwork( lgraph );
            
        end


        function [ loss, state ] = calcLoss(  self, net, dlX, dlC )
            % Calculate the classifier loss
            arguments
                self     InputClassifierLoss
                net      dlnetwork
                dlX      dlarray  % input distribtion
                dlC      dlarray  % class distribution
            end

            if self.HasNetwork                    
                [ loss, state ] = self.networkLoss( net, dlX, dlC );
            
            else
                loss = self.nonNetworkLoss( self.ModelType, dlX, dlC );
                state = [];

            end

    
        end


    end

    methods (Access = protected)

        function [ loss, state ] = networkLoss( self, net, dlX, dlC )

            if size( dlX, 3 ) > 1 && net.Layers(1).InputSize > 1
                % flatten the input
                dlX = flattenDLArray( dlX );
            end

            % get the network's predicted class
            [ dlCGen, state ] = forward( net, dlX );

            % hotcode the actual class 
            dlCActual = dlarray( ...
                onehotencode( self.CLabels(dlC), 1 ), 'CB' );
            
            % compute the cross entropy (classifier) loss
            loss = crossentropy( dlCGen, dlCActual, ...
                                  'TargetCategories', 'Independent' );

        end


        function loss = nonNetworkLoss( self, dlX, dlC )

            % convert to double for models which don't take dlarrays
            X = double(extractdata( dlX ))';
            C = double(extractdata( dlC ));
            
            % fit the appropriate model
            switch self.ModelType
                case 'Logistic'
                    model = fitclinear( X, C, Learner = "logistic" );
                case 'Fisher'
                    model = fitcdiscr( X, C );
                case 'SVM'
                    model = fitcecoc( X, C );
            end
            
            % compute the training loss
            loss = resubLoss( model );
        
        end

    end

end
