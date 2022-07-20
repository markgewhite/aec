classdef ModelOptimizer < handle
    % Class defining the optimizer for the model's networks

    properties
        Type            % type of optimizer
        NetNames        % list network names for convenience
        LearningRates   % learning rates for all networks
        States          % optimizer states for all networks
        LRFactor        % learning rate reduction factor 
    end

    methods

        function self = ModelOptimizer( netNames, args )
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
            self.NetNames = netNames;
            self.Type = args.type;
            self.LRFactor = args.lrFactor;

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
                    self.LearningRates.(netNames{i}) = args.initLearningRates(i);
                else
                    self.LearningRates.(netNames{i}) = args.initLearningRates;
                end
                switch self.Type
                    case 'ADAM'
                        self.States.(netNames{i}).AvgG = []; 
                        self.States.(netNames{i}).AvgGS = [];
                        self.States.(netNames{i}).Beta1 = args.beta1;
                        self.States.(netNames{i}).Beta2 = args.beta2;
                    case 'SGDM'
                        self.States.(netNames{i}).vel = [];
                end
            end


        end


        function nets = updateNets( self, nets, ...
                                    grads, count )
            % Update the network parameters
            arguments
                self            ModelOptimizer
                nets            struct
                grads           struct
                count           double
            end

            nNets = length( self.NetNames );
            for i = 1:nNets

                thisName = self.NetNames{i};

                if ~isfield( grads, thisName )
                    % skip as no gradient information
                    continue
                end

                thisState = self.States.(thisName);
                thisLearningRate = self.LearningRates.(thisName);
                % update the network parameters
                switch self.Type
                    case 'ADAM'         
                        [ nets.(thisName), ...
                          thisState.AvgG, ...
                          thisState.AvgGS ] = ...
                                adamupdate( nets.(thisName), ...
                                            grads.(thisName), ...
                                            thisState.AvgG, ...
                                            thisState.AvgGS, ...
                                            count, ...
                                            thisLearningRate, ...
                                            thisState.Beta1, ...
                                            thisState.Beta2 );
                    case 'SGD'
                        [ nets.(thisName), ...
                          thisState.vel ] = ...
                            sgdmupdate( nets.(thisName), ...
                                        grads.(thisName), ...
                                        thisState.vel, ...
                                        thisLearningRate );
                end
                
                self.States.(thisName) = thisState;
            
            end

        end


        function self = updateLearningRates( self, preTraining )
            % Update learning rates
            arguments
                self            ModelOptimizer
                preTraining     logical
            end

            for i = 1:length( self.NetNames )

                thisName = self.NetNames{i};
                if preTraining ...
                    && ~any(strcmp( thisName, {'Encoder','Decoder'} )) 
                    % skip update of any other networks if pretraining
                    continue
                end

                self.LearningRates.(thisName) = ...
                            self.LearningRates.(thisName)*self.LRFactor;
            end

        end
          
    end


end


