% ************************************************************************
% Class: adversarialLoss
%
% Subclass for adversarial loss using a discriminator network
%
% ************************************************************************

classdef adversarialLoss < lossFcn

    properties
        discriminator         % dlnetwork object
        initLearningRate      % initial learning rate
        type                  % type of target distribution
    end

    methods

        function self = adversarialLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name            char {mustBeText}
                superArgs.?lossFcn
                args.type       char ...
                    {mustBeMember( args.type, ...
                                    {'Gaussian', ...
                                     'Categorical'} )} = 'Gaussian'
                args.ZDim       double ...
                            {mustBeInteger, mustBePositive} = 4
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
            self = self@lossFcn( name, superArgsCell{:}, ...
                                 type = 'Regularization' );
            self.initLearningRate = args.initLearningRate;

            % create the input layer
            layers = featureInputLayer( args.ZDim, 'Name', 'in' );
            
            % create the hidden layers
            for i = 1:args.nHidden
                nNodes = fix( args.nFC*2^(args.fcFactor*(1-i)) );
                layers = [ layers; ...
                    fullyConnectedLayer( nNodes, 'Name', ['fc' num2str(i)] )
                    batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
                    leakyReluLayer( args.scale, 'Name', ['relu' num2str(i)] )
                    dropoutLayer( args.dropout, 'Name', ['drop' num2str(i)] )
                    ]; %#ok<AGROW> 
            end
            
            % create final layers
            layers = [ layers; ...    
                            fullyConnectedLayer( 1, 'Name', 'fcout' )
                            sigmoidLayer( 'Name', 'out' )
                            ];
            
            lgraph = layerGraph( layers );
            self.discriminator = dlnetwork( lgraph );


        end

    end

    methods (Static)

        function [ loss, state ] = calcLoss( self, dlZFake )
            % Calculate the adversarial loss
            arguments
                self
                dlZFake  dlarray  % generated distribution
            end

            if self.doCalcLoss

                [ ZDim, batchSize ] = size( dlZFake );

                % generate a target distribution
                switch self.type
                    case 'Gaussian'
                        dlZReal = dlarray( randn( ZDim, batchSize ), 'CB' );
                    case 'Categorical'
                        dlZReal = dlarray( randi( ZDim, batchSize ), 'CB' );
                end

                % predict authenticity from real Z using the discriminator
                dlDReal = forward( self.discriminator, dlZReal );
                
                % predict authenticity from fake Z
                [ dlDFake, state ] = forward( self.discriminator, dlZFake );
                
                % discriminator loss
                loss(1) = 0.5*mean( log(dlDReal + eps) + log(1 - dlDFake + eps) );
                % generator loss
                loss(2) = mean( log(dlDFake + eps) );
            
            else
                loss = [0 0];
                state = [];
            end
    
        end

    end

end
