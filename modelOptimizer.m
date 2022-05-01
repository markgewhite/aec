classdef modelOptimizer 
    % Class defining the optimizer for the model's networks

    properties
        type            % type of optimizer
        netNames        % list network names for convenience
        learningRates   % learning rates for all networks
        states          % optimizer states for all networks
        lrFactor        % learning rate reduction factor 
    end

    methods

        function self = modelOptimizer( netNames, args )
            % Initialize the model
            arguments
                netNames            string ...
                    {mustBeText};
                args.type           char ...
                    {mustBeMember(args.type, {'ADAM', 'SGDM'} )} = 'ADAM';
                args.beta1          double ...
                    {mustBeNumeric, mustBePositive} = 0.9;
                args.beta2          double ...
                    {mustBeNumeric, mustBePositive} = 0.999;
                args.initLearningRates  double ...
                    {mustBePositive} = 0.001;
                args.lrFactor       double ...
                    {mustBeNumeric, mustBePositive} = 0.5;
            end

            % initialize the optimization parameters
            self.netNames = netNames;
            self.type = args.type;
            self.lrFactor = args.lrFactor;

            nNetworks = length( netNames );

            if length( args.initLearningRates ) > 1
                if length( args.initLearningRates ) == nNetworks
                    netSpecific = true;
                else
                    eid = 'Autoencoder:LearningRateMisMatch';
                    msg = 'The number of learning rates does not match the number of networks.';
                    throwAsCaller( MException(eid,msg) );
                end
            else
                netSpecific = false;
            end   
            
            for i = 1:nNetworks
                if netSpecific
                    self.learningRates.(netNames{i}) = args.initLearningRates(i);
                else
                    self.learningRates.(netNames{i}) = args.initLearningRates;
                end
                switch self.type
                    case 'ADAM'
                        self.states.(netNames{i}).avgG = []; 
                        self.states.(netNames{i}).avgGS = [];
                        self.states.(netNames{i}).beta1 = args.beta1;
                        self.states.(netNames{i}).beta2 = args.beta2;
                    case 'SGDM'
                        self.states.(netNames{i}).vel = [];
                end
            end


        end


        function [ self, nets ] = updateNets( self, nets, ...
                                    grads, count, doTrainAE )
            % Update the network parameters
            arguments
                self        modelOptimizer
                nets        struct
                grads       struct
                count       double
                doTrainAE   logical
            end

            nNets = length( self.netNames );
            for i = 1:nNets

                thisName = self.netNames{i};
                if any(strcmp( thisName, {'encoder','decoder'} )) ...
                    && not(doTrainAE)
                    % skip training for the AE
                    continue
                end

                if ~isfield( grads, thisName )
                    % skip as no gradient information
                    continue
                end

                thisState = self.states.(thisName);
                thisLearningRate = self.learningRates.(thisName);
                % update the network parameters
                switch self.type
                    case 'ADAM'         
                        [ nets.(thisName), ...
                          thisState.avgG, ...
                          thisState.avgGS ] = ...
                                adamupdate( nets.(thisName), ...
                                            grads.(thisName), ...
                                            thisState.avgG, ...
                                            thisState.avgGS, ...
                                            count, ...
                                            thisLearningRate, ...
                                            thisState.beta1, ...
                                            thisState.beta2 );
                    case 'SGD'
                        [ nets.(thisName), ...
                          thisState.vel ] = ...
                            sgdmupdate( nets.(thisName), ...
                                        grads.(thisName), ...
                                        thisState.vel, ...
                                        thisLearningRate );
                end
                
                self.states.(thisName) = thisState;
            
            end

        end


        function self = updateLearningRates( self, doTrainAE )
            % Update learning rates
            arguments
                self         modelOptimizer
                doTrainAE    logical
            end

            for i = 1:length( self.netNames )

                thisName = self.netNames{i};
                if any(strcmp( thisName, {'encoder','decoder'} )) ...
                    && not(doTrainAE)
                    % skip training for the AE
                    continue
                end

                self.learningRates.(thisName) = ...
                            self.learningRates.(thisName)*self.lrFactor;
            end

        end
          
    end


end


