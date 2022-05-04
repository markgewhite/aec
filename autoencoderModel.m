% ************************************************************************
% Class: autoencoderModel
%
% Subclass defining the framework for an autoencoder model
%
% ************************************************************************

classdef autoencoderModel < representationModel

    properties
        XOutDim        % output dimension (may differ from XDim)
        nets           % networks defined in this model (structure)
        netNames       % names of the networks (for convenience)
        nNets          % number of networks
        isVAE          % flag indicating if variational autoencoder
        lossFcns       % array of loss functions
        lossFcnNames   % names of the loss functions
        lossFcnWeights % weights to be applied to the loss function
        lossFcnTbl     % convenient table summarising loss function details
        nLoss          % number of computed losses
        isInitialized  % flag indicating if fully initialized
        hasSeqInput    % supports variable-length input
        auxNetwork     % name of auxiliary dlnetwork
        auxModelType   % type of auxiliary model to use
        auxModel       % auxiliary model itself

        trainer        % trainer object holding training parameters
        optimizer      % optimizer object


    end

    methods

        function self = autoencoderModel( XDim, ...
                                          XChannels, ...
                                          lossFcns, ...
                                          superArgs, ...
                                          args )
            % Initialize the model
            arguments
                XDim            double {mustBeInteger, mustBePositive}
                XChannels       double {mustBeInteger, mustBePositive}
            end
            arguments (Repeating)
                lossFcns      lossFunction
            end
            arguments
                superArgs.?representationModel
                args.XOutDim        double = 0
                args.hasSeqInput    logical = false
                args.isVAE          logical = false
                args.weights        double ...
                                    {mustBeNumeric,mustBeVector} = 1
                args.auxModel       string ...
                        {mustBeMember( args.auxModel, ...
                                {'Fisher', 'SVM'} )} = 'Fisher'
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@representationModel( superArgsCell{:}, ...
                                             NumCompLines = 8 );

            self.XDim = XDim;
            self.XChannels = XChannels;

            % placeholders for subclasses to define
            self.nets.encoder = [];
            self.nets.decoder = [];
            self.netNames = {'encoder', 'decoder'};
            self.nNets = 2;
            self.isVAE = args.isVAE;
            self.hasSeqInput = args.hasSeqInput;

            self.auxModelType = args.auxModel;
            self.auxModel = [];

            if args.XOutDim == 0
                self.XOutDim = self.XDim;
            end

            % copy over the loss functions associated
            % and any networks with them for later training 
            self = addLossFcns( self, lossFcns{:}, weights = args.weights );

            self.nLoss = sum( self.lossFcnTbl.nLosses );

            % indicate that further initialization is required
            self.isInitialized = false;

        end


        function self = addLossFcns( self, newFcns, args )
            % Add one or more loss function objects to the model
            arguments
                self
            end
            arguments (Repeating)
                newFcns   lossFunction
            end
            arguments
                args.weights double {mustBeNumeric,mustBeVector} = 1
            end
       
            nFcns = length( newFcns );

            % check the weights
            if args.weights==1
                % default is to assign a weight of 1 to all functions
                w = ones( nFcns, 1 );
            elseif length( args.weights ) ~= nFcns
                % weights don't correspond to the functions
                eid = 'aeModel:WeightsMismatch';
                msg = 'Number of assigned weights does not match number of functions';
                throwAsCaller( MException(eid,msg) );
            else
                w = args.weights;
            end
            self.lossFcnWeights = [ self.lossFcnWeights w ];

            % update the names list
            self.lossFcnNames = [ self.lossFcnNames getFcnNames(newFcns) ];
            % add to the loss functions
            for i = 1:length( newFcns )
                self.lossFcns.(newFcns{i}.name) = newFcns{i};
            end

            % add networks, if required
            self = addLossFcnNetworks( self, newFcns );
          
            % store the loss functions' details 
            % and relevant details for easier access when training
            self = self.setLossInfoTbl;
            self.lossFcnTbl.types = categorical( self.lossFcnTbl.types );
            self.lossFcnTbl.inputs = categorical( self.lossFcnTbl.inputs );

            % check a reconstruction loss is present
            if ~any( self.lossFcnTbl.types=='Reconstruction' )
                eid = 'aeModel:NoReconstructionLoss';
                msg = 'No reconstruction loss object has been specified.';
                throwAsCaller( MException(eid,msg) );
            end 

            % check there is no more than one auxiliary network, if at all
            auxFcns = self.lossFcnTbl.types=='Auxiliary';
            if sum( auxFcns ) > 1
                eid = 'aeModel:MulitpleAuxiliaryFunction';
                msg = 'There is more than one auxiliary loss function.';
                throwAsCaller( MException(eid,msg) );
            elseif sum( auxFcns ) == 1
                % for convenience identify the auxiliary network
                auxNet = (self.netNames == self.lossFcnNames(auxFcns));
                self.auxNetwork = self.netNames( auxNet );
            end

        end


        function self = initTrainer( self, args )
            arguments
                self
                args.?modelTrainer
            end

            % set the trainer's properties
            argsCell = namedargs2cell( args );
            self.trainer = modelTrainer( self.lossFcnTbl, ...
                                         argsCell{:}, ...
                                         showPlots = self.ShowPlots );

            % confirm if initialization is complete
            self.isInitialized = ~isempty(self.optimizer);

        end
        

        function self = initOptimizer( self, args )
            arguments
                self
                args.?modelOptimizer
            end

            % set the trainer's properties
            argsCell = namedargs2cell( args );
            self.optimizer = modelOptimizer( self.netNames, argsCell{:} );

            % confirm if initialization is complete
            self.isInitialized = ~isempty(self.trainer);

        end


        function self = train( self, thisDataset )
            % Train the autoencoder model
            arguments
                self          autoencoderModel
                thisDataset   modelDataset
            end

            if ~self.isInitialized
                eid = 'Autoencoder:NotInitialized';
                msg = 'The trainer, optimizer or dataset parameters have not all been set.';
                throwAsCaller( MException(eid,msg) );
            end

            % check dataset is suitable
            if thisDataset.isFixedLength == self.hasSeqInput
                eid = 'aeModel:DatasetNotSuitable';
                if thisDataset.isFixedLength
                    msg = 'The dataset should have variable length for the model.';
                else
                    msg = 'The dataset should have fixed length for the model.';
                end
                throwAsCaller( MException(eid,msg) );
            end

            % re-partition the data to create training and validation sets
            cvPart = cvpartition( thisDataset.nObs, 'Holdout', 0.25 );
            
            thisTrnSet = thisDataset.partition( training(cvPart) );
            thisValSet = thisDataset.partition( test(cvPart) );

            % run the training loop
            [ self, self.optimizer ] = ...
                            self.trainer.runTraining( self, ...
                                                      thisTrnSet, ...
                                                      thisValSet );


        end


        function loss = getReconLoss( self, X, XHat )
            % Calculate the reconstruction loss
            arguments
                self
                X     
                XHat  
            end
            
            name = self.lossFcnTbl.names( self.lossFcnTbl.types=='Reconstruction' );
            loss = self.lossFcns.(name).calcLoss( X, XHat );

        end


        function dlXC = latentComponents( self, dlZ, args )
            % Calculate the funtional components from the latent codes
            % using the decoder network. For each component, the relevant 
            % code is varied randomly about the mean. This is more 
            % efficient than calculating two components at strict
            % 2SD separation from the mean.
            arguments
                self            autoencoderModel
                dlZ             
                args.sampling   char ...
                    {mustBeMember(args.sampling, ...
                        {'Random', 'Fixed'} )} = 'Random'
                args.centre     logical = true
                args.nSample    double {mustBeInteger} = 0
                args.range      double {mustBePositive} = 2.0
            end

            if args.nSample > 0
                nSample = args.nSample;
            else
                nSample = self.NumCompLines;
            end

            if ~isa( dlZ, 'dlarray' )
                dlZ = dlarray( dlZ', 'CB' );
            end

            ZDim = size( dlZ, 1 );
            
            % compute the mean and SD across the batch
            dlZMean = mean( dlZ, 2 );
            dlZStd = std( dlZ, [], 2 );
            
            % initialise the components' Z codes at the mean
            % include an extra one that will be preserved
            dlZC = repmat( dlZMean, 1, ZDim*nSample+1 );

            % convert to double to speed up processing
            % provided dlZC can trace back to original dlZ
            ZMean = double(extractdata( dlZMean ));
            ZStd = double(extractdata( dlZStd ));
            
            if strcmp( args.sampling, 'Fixed' )
                % define the offset spacing
                offsets = linspace( -args.range, args.range, nSample );
            end

            for j = 1:nSample
                
                switch args.sampling
                    case 'Random'
                        offset = args.range*rand;
                    case 'Fixed'
                        offset = offsets(j);
                end

                for i =1:ZDim
                    % vary ith component randomly about its mean
                    dlZC(i,(i-1)*nSample+j) = ZMean(i) + offset*ZStd(i);
                end

            end

            % generate all the component curves using the decoder
            dlXC = forward( self.nets.decoder, dlZC );

            XDim = size( dlXC );

            if args.centre
                % centre about the mean curve (last curve)
                if length( XDim )==2
                    dlXC = dlXC( :, 1:end-1 ) - dlXC( :, end );
                else
                    dlXC = dlXC( :, :, 1:end-1 ) - dlXC( :, :, end );
                end
            end

        end
    
    
    end


    methods (Access = protected)


        function self = addLossFcnNetworks( self, newFcns )
            % Add one or more networks to the model
            arguments
                self
                newFcns
            end

            nFcns = length( newFcns );
            k = length( self.nets );
            for i = 1:nFcns
                thisLossFcn = newFcns{i};
                if thisLossFcn.hasNetwork
                    k = k+1;
                    % add the network object
                    self.nets.(thisLossFcn.name) = ...
                                initNetwork( thisLossFcn, self.ZDim );
                    % record its name
                    self.netNames = [ string(self.netNames) thisLossFcn.name ];
                    % increment the counter
                    self.nNets = self.nNets + 1;
                end
            end


        end

        function self = setLossInfoTbl( self )
            % Update the info table
            
            nFcns = length( self.lossFcnNames );
            names = strings( nFcns, 1 );
            types = strings( nFcns, 1 );
            inputs = strings( nFcns, 1 );
            weights = zeros( nFcns, 1 );
            nLosses = zeros( nFcns, 1 );
            lossNets = strings( nFcns, 1 );
            hasNetwork = false( nFcns, 1 );
            doCalcLoss = false( nFcns, 1 );
            useLoss = false( nFcns, 1 );

            for i = 1:nFcns
                
                thisLossFcn = self.lossFcns.(self.lossFcnNames(i));
                names(i) = thisLossFcn.name;
                types(i) = thisLossFcn.type;
                inputs(i) = thisLossFcn.input;
                weights(i) = self.lossFcnWeights(i);
                nLosses(i) = thisLossFcn.nLoss;
                hasNetwork(i) = thisLossFcn.hasNetwork;
                doCalcLoss(i) = thisLossFcn.doCalcLoss;
                useLoss(i) = thisLossFcn.useLoss;

                nFcnNets = length( thisLossFcn.lossNets );
                for j = 1:nFcnNets
                    if length(string( thisLossFcn.lossNets{j} ))==1
                        assignments = thisLossFcn.lossNets{j};
                    else
                        assignments = strjoin( thisLossFcn.lossNets{j,:}, '+' );
                    end
                    lossNets(i) = strcat( lossNets(i), assignments ) ;
                    if j < nFcnNets
                        lossNets(i) = strcat( lossNets(i), "; " );
                    end
                end

            end

            self.lossFcnTbl = table( names, types, inputs, weights, ...
                    nLosses, lossNets, hasNetwork, doCalcLoss, useLoss );

        end


    end


    methods (Static)

        function dlZ = encode( self, X, arg )
            % Encode features Z from X using the model
            arguments
                self            autoencoderModel
                X
                arg.convert     logical = true
            end

            if isa( X, 'modelDataset' )
                dlX = X.getInput;
            elseif isa( X, 'dlarray' )
                dlX = X;
            else
                eid = 'Autoencoder:NotValidX';
                msg = 'The input data should be a modelDataset or a dlarray.';
                throwAsCaller( MException(eid,msg) );
            end

            dlZ = predict( self.nets.encoder, dlX );

            if arg.convert
                dlZ = double(extractdata( dlZ ))';
            end

        end


        function dlXHat = reconstruct( self, Z, arg )
            % Reconstruct X from Z using the model
            arguments
                self            autoencoderModel
                Z
                arg.convert     logical = true
            end

            if isa( Z, 'dlarray' )
                dlZ = Z;
            else
                dlZ = dlarray( Z', 'CB' );
            end

            dlXHat = predict( self.nets.decoder, dlZ );

            if arg.convert
                dlXHat = double(extractdata( dlXHat ));
            end
            

        end


        function YHat = predictAux( self, Z )
            % Make prediction from X or Z using an auxiliary network
            arguments
                self            autoencoderModel
                Z
            end

            if isa( Z, 'dlarray' )
                dlZ = Z;
            else
                dlZ = dlarray( Z, 'CB' );
            end

            auxNet = (self.lossFcnTbl.types == 'Auxiliary');
            if ~any( auxNet )
                eid = 'aeModel:NoAuxiliaryFunction';
                msg = 'No auxiliary loss function specified in the model.';
                throwAsCaller( MException(eid,msg) );
            end

            if ~self.lossFcnTbl.hasNetwork( auxNet )
                eid = 'aeModel:NoAuxiliaryNetwork';
                msg = 'No auxiliary network specified in the model.';
                throwAsCaller( MException(eid,msg) );
            end

            auxNetName = self.lossFcnTbl.names( auxNet );
            dlYHat = predict( self.nets.(auxNetName), dlZ );

            YHat = double(extractdata( dlYHat ));

        end


        function net = getNetwork( self, name )
            arguments
                self
                name         string {mustBeNetName( self, name )}
            end
            
            net = self.nets.(name);
            if isa( net, 'lossFunction' )
                net = self.lossFcns.(name).net;
            end

        end

        
        function names = getNetworkNames( self )
            arguments
                self
            end
            
            names = fieldnames( self.nets );

        end


        function isValid = mustBeNetName( self, name )
            arguments
                self
                name
            end

            isValid = ismember( name, self.names );

        end


        function [ dlXHat, dlZ, state ] = forward( encoder, decoder, dlX )
            % Forward-run the autoencoder networks
            arguments
                encoder     dlnetwork
                decoder     dlnetwork
                dlX         dlarray
            end

            % generate latent encodings
            [ dlZ, state.encoder ] = forward( encoder, dlX );
    
            % reconstruct curves from latent codes
            [ dlXHat, state.decoder ] = forward( decoder, dlZ );

        end


        function [ dlYHat, state ] = forwardAux( auxNet, dlZ )
            % Forward-run the auxiliary network
            arguments
                auxNet          dlnetwork
                dlZ             dlarray
            end

            [ dlYHat, state] = forward( auxNet, dlZ );

        end


    end


end


function names = getFcnNames( lossFcns )

    nFcns = length( lossFcns );
    names = strings( nFcns, 1 );
    for i = 1:nFcns
        names(i) = lossFcns{i}.name;
    end

end



