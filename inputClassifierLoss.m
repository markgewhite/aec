% ************************************************************************
% Class: inputClassifierLoss
%
% Subclass for classifier loss with X as the input for comparison purposes
%
% ************************************************************************

classdef inputClassifierLoss < lossFunction

    properties
        XDim                % input dimension size
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

        function self = inputClassifierLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name            char {mustBeText}
                superArgs.?lossFunction
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

            self = self@lossFunction( name, superArgsCell{:}, ...
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
                self            inputClassifierLoss
                thisModel       autoencoderModel
            end

            self.XDim = thisModel.XDim;
            self.CDim = thisModel.CDim;
            self.CLabels = categorical( 1:self.CDim );

        end
        
        
        function net = initNetwork( self )
            % Generate an initialized network
            arguments
                self       inputClassifierLoss
            end

            if ~strcmp( self.ModelType, 'Network' )
                eid = 'classifierLoss:NotNetwork';
                msg = 'This classifier function does not use a network.';
                throwAsCaller( MException(eid,msg) );
            end 

            % create the input layer
            layers = featureInputLayer( self.XDim, ...
                                   'Name', 'in', ...
                                   'Normalization', 'zscore', ...
                                   'Mean', 0, 'StandardDeviation', 1 );
            
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
                            sigmoidLayer( 'Name', 'out' )
                            ];
            
            lgraph = layerGraph( layers );
            net = dlnetwork( lgraph );
            
        end


        function [ loss, state ] = calcLoss(  self, net, dlX, dlC )
            % Calculate the classifier loss
            arguments
                self     inputClassifierLoss
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