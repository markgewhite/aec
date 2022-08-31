classdef AdversarialLoss < LossFunction
    % Subclass for adversarial loss using a discriminator network

    properties
        ZDim                % latent codes dimension size  
        NumHidden           % number of hidden layers
        NumFC               % number of fully connected nodes at widest
        FCFactor            % node ratio specifying the power 2 index
        Scale               % leaky Relu scale
        Dropout             % dropout rate
        InitLearningRate    % initial learning rate
        Distribution        % type of target distribution
    end

    methods

        function self = AdversarialLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name            char {mustBeText}
                superArgs.?LossFunction
                args.distribution  char ...
                    {mustBeMember( args.distribution, ...
                                    {'Gaussian', ...
                                     'DoubleGaussian', ...
                                     'Categorical'} )} = 'Gaussian'
                args.numHidden  double ...
                            {mustBeInteger, mustBePositive} = 3
                args.numFC      double ...
                            {mustBeInteger, mustBePositive} = 256
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
            netAssignments = { string(name); "Encoder" };

            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Regularization', ...
                                 input = 'Z', ...
                                 nLoss = 2, ...
                                 lossNets = netAssignments, ...
                                 hasNetwork = true, ...
                                 hasState = true );

            self.NumHidden = args.numHidden;
            self.NumFC = args.numFC;
            self.FCFactor = args.fcFactor;
            self.Scale = args.scale;
            self.Dropout = args.dropout;
            
            self.Distribution = args.distribution;
            self.InitLearningRate = args.initLearningRate;

        end


        function self = setDimensions( self, thisModel )
            % Store the Z dimension input once known
            arguments
                self            AdversarialLoss
                thisModel       FullAEModel
            end

            self.ZDim = thisModel.ZDim;

        end

        
        function net = initNetwork( self )
            % Generate an initialized network
            arguments
                self
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
                            fullyConnectedLayer( 1, 'Name', 'fcout' )
                            softmaxLayer( 'Name', 'out' )
                            ];
            
            lgraph = layerGraph( layers );
            net = dlnetwork( lgraph );


        end


        function [ loss, state ] = calcLoss( self, net, dlZFake )
            % Calculate the adversarial loss
            arguments
                self     AdversarialLoss
                net      dlnetwork
                dlZFake  dlarray  % generated distribution
            end

            [ ZSize, batchSize ] = size( dlZFake );
            if ZSize ~= self.ZDim
                eid = 'adversarilLoss:IncorrectZ';
                msg = 'Input array has the wrong number of latent codes.';
                throwAsCaller( MException(eid,msg) );
            end 

            % generate a target distribution
            switch self.Distribution
                case 'Gaussian'
                    dlZReal = dlarray( randn( ZSize, batchSize ), 'CB' );
                case 'DoubleGaussian'
                    dlZReal = dlarray( randn( ZSize, batchSize ), 'CB' );
                    dlZReal(:,1:fix(batchSize/2)) = ...
                                    dlZReal(:,1:fix(batchSize/2)) - 2;
                    dlZReal(:,fix(batchSize/2)+1:end) = ...
                                dlZReal(:,fix(batchSize/2)+1:end) + 2;

                case 'Categorical'
                    dlZReal = dlarray( randi( ZSize, batchSize ), 'CB' );
            end

            % predict authenticity from real Z using the discriminator
            dlDReal = forward( net, dlZReal );
            
            % predict authenticity from fake Z
            [ dlDFake, state ] = forward( net, dlZFake );
            
            % discriminator loss
            loss(1) = -0.5*mean( log(dlDReal + eps) + log(1 - dlDFake + eps) );
            % generator loss
            loss(2) = -mean( log(dlDFake + eps) );

    
        end

    end

end
