% ************************************************************************
% Class: adversarialLoss
%
% Subclass for adversarial loss using a discriminator network
%
% ************************************************************************

classdef adversarialLoss < lossFunction

    properties
        ZDim                % latent codes dimension size  
        nHidden             % number of hidden layers
        nFC                 % number of fully connected nodes at widest
        fcFactor            % node ratio specifying the power 2 index
        scale               % leaky Relu scale
        dropout             % dropout rate
        initLearningRate    % initial learning rate
        distribution        % type of target distribution
    end

    methods

        function self = adversarialLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name            char {mustBeText}
                superArgs.?lossFunction
                args.distribution  char ...
                    {mustBeMember( args.distribution, ...
                                    {'Gaussian', ...
                                     'Categorical'} )} = 'Gaussian'
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
            netAssignments = { string({'decoder',name}); {'encoder'} };

            self = self@lossFunction( name, superArgsCell{:}, ...
                                 type = 'Regularization', ...
                                 input = 'Z', ...
                                 nLoss = 2, ...
                                 lossNets = netAssignments, ...
                                 hasNetwork = true, ...
                                 hasState = true );

            self.nHidden = args.nHidden;
            self.nFC = args.nFC;
            self.fcFactor = args.fcFactor;
            self.scale = args.scale;
            self.dropout = args.dropout;
            
            self.distribution = args.distribution;
            self.initLearningRate = args.initLearningRate;

        end


        function net = initNetwork( self, ZDim )
            % Generate an initialized network
            arguments
                self
                ZDim       double {mustBeInteger, mustBePositive}
            end

            self.ZDim = ZDim;

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
                            fullyConnectedLayer( 1, 'Name', 'fcout' )
                            sigmoidLayer( 'Name', 'out' )
                            ];
            
            lgraph = layerGraph( layers );
            net = dlnetwork( lgraph );


        end


        function [ loss, state ] = calcLoss( self, net, dlZFake )
            % Calculate the adversarial loss
            arguments
                self     adversarialLoss
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
            switch self.distribution
                case 'Gaussian'
                    dlZReal = dlarray( randn( ZSize, batchSize ), 'CB' );
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
