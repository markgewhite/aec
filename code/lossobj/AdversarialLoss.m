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
                args.Distribution  char ...
                    {mustBeMember( args.Distribution, ...
                                    {'Gaussian', ...
                                     'DoubleGaussian', ...
                                     'Cauchy', ...
                                     'Categorical'} )} = 'Gaussian'
                args.NumHidden  double ...
                            {mustBeInteger, mustBePositive} = 2
                args.NumFC      double ...
                            {mustBeInteger, mustBePositive} = 64
                args.FCFactor   double ...
                            {mustBeInteger, mustBePositive} = 2
                args.Scale      double ...
                            {mustBeInRange(args.Scale, 0, 1)} = 0.2
                args.Dropout    double ...
                            {mustBeInRange(args.Dropout, 0, 1)} = 0.1
                args.InitLearningRate     double ...
                    {mustBeInRange(args.InitLearningRate, 0, 1)} = 0.001
            end

            superArgsCell = namedargs2cell( superArgs );
            netAssignments = { string(name); "Encoder" };

            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Regularization', ...
                                 input = {'dlZ'}, ...
                                 nLoss = 2, ...
                                 lossNets = netAssignments, ...
                                 hasNetwork = true, ...
                                 hasState = true );

            self.NumHidden = args.NumHidden;
            self.NumFC = args.NumFC;
            self.FCFactor = args.FCFactor;
            self.Scale = args.Scale;
            self.Dropout = args.Dropout;
            
            self.Distribution = args.Distribution;
            self.InitLearningRate = args.InitLearningRate;

        end


        function self = setDimensions( self, thisModel )
            % Store the Z dimension input once known
            arguments
                self            AdversarialLoss
                thisModel       AEModel
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
                            sigmoidLayer( 'Name', 'out' )
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

                case 'Cauchy'
                    dlZReal = dlarray( trnd( 1, ZSize, batchSize ), 'CB' );

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
