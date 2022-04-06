% ************************************************************************
% Class: classifierLoss
%
% Subclass for classifier loss using an auxiliary network
%
% ************************************************************************

classdef classifierLoss < lossFcn

    properties
        classifier         % dlnetwork object
        initLearningRate   % initial learning rate
        CDim               % number of possible classes
        type               % type of classifier model
        isNetwork          % flag indicating if network defined
    end

    methods

        function self = classifierLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name            char {mustBeText}
                superArgs.?lossFcn
                args.type       char ...
                    {mustBeMember( args.type, ...
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
            self = self@lossFcn( name, superArgsCell{:}, ...
                                 type = 'Classification' );
            self.CDim = args.CDim;
            self.type = args.type;

            switch args.type
                case 'Network'
                    self.isNetwork = true;
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
                                    fullyConnectedLayer( args.CDim, 'Name', 'fcout' )
                                    sigmoidLayer( 'Name', 'out' )
                                    ];
                    
                    lgraph = layerGraph( layers );
                    self.classifier = dlnetwork( lgraph );

                otherwise
                    % not a network, will use a fixed model
                    self.classifier = [];
                    self.isNetwork = false;
                    self.initLearningRate = 0;

            end


        end

    end

    methods (Static)

        function [ loss, state ] = calcLoss( self, dlZGen, dlC )
            % Calculate the classifier loss
            arguments
                self
                dlZGen  dlarray  % generated latent distribtion
                dlC     dlarray  % actual distribution
            end

            if self.doCalcLoss

                if self.isNetwork                    
                    [ loss, state ] = ...
                        networkLoss( self.classifier, dlZGen, dlC );
                else

                    loss = nonNetworkLoss( self.type, dlZGen, dlC );
                    state = [];

                end

            else

                loss = 0;
                state = [];
            end
    
        end


    end

    methods (Access = protected)

        function [ loss, state ] = networkLoss( model, dlZGen, dlC )

            % get the network's predicted class
            [ dlCGen, state ] = forward( model, dlZGen );

            % hotcode the actual class 
            dlCActual = dlarray( ...
                onehotencode( self.cLabels(dlC+1), 1 ), 'CB' );
            
            % compute the cross entropy (classifier) loss
            loss = crossentropy( dlCGen, dlCActual, ...
                                  'TargetCategories', 'Independent' );

        end


        function loss = nonNetworkLoss( model, dlZGen, dlC )

            % convert to double for models which don't take dlarrays
            ZGen = double(extractdata( dlZGen ))';
            C = double(extractdata( dlC ));
            
            % fit the appropriate model
            switch model
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
