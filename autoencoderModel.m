% ************************************************************************
% Class: autoencoderModel
%
% Subclass defining the framework for an autoencoder model
%
% ************************************************************************

classdef autoencoderModel < representationModel

    properties
        XOutputDim     % output dimension (may differ from XInputDim)
        Nets           % networks defined in this model (structure)
        NetNames       % names of the networks (for convenience)
        NumNetworks    % number of networks
        IsVAE          % flag indicating if variational autoencoder
        NumVAEDraws    % number of draws from encoder output distribution
        LossFcns       % array of loss functions
        LossFcnNames   % names of the loss functions
        LossFcnWeights % weights to be applied to the loss function
        LossFcnTbl     % convenient table summarising loss function details
        NumLoss        % number of computed losses
        IsInitialized  % flag indicating if fully initialized
        HasSeqInput    % supports variable-length input
        AuxNetwork     % name of auxiliary dlnetwork
        AuxModelType   % type of auxiliary model to use
        AuxModel       % auxiliary model itself

        Trainer        % trainer object holding training parameters
        Optimizer      % optimizer object
    end

    properties (Dependent = true)
        XDimLabels     % dimensional labelling for X input dlarrays
        XNDimLabels    % dimensional labelling for time-normalized output
    end

    methods

        function self = autoencoderModel( XDim, ...
                                          XOutputDim, ...
                                          XChannels, ...
                                          ZDim, ...
                                          CDim, ...
                                          lossFcns, ...
                                          superArgs, ...
                                          args )
            % Initialize the model
            arguments
                XDim            double {mustBeInteger, mustBePositive}
                XOutputDim      double {mustBeInteger, mustBePositive}
                XChannels       double {mustBeInteger, mustBePositive}
                ZDim            double {mustBeInteger, mustBePositive}
                CDim            double {mustBeInteger, mustBePositive}
            end
            arguments (Repeating)
                lossFcns      lossFunction
            end
            arguments
                superArgs.?representationModel
                args.hasSeqInput    logical = false
                args.isVAE          logical = false
                args.numVAEDraws    double ...
                    {mustBeInteger, mustBePositive} = 1
                args.weights        double ...
                                    {mustBeNumeric,mustBeVector} = 1
                args.auxModel       string ...
                        {mustBeMember( args.auxModel, ...
                           {'Logistic', 'Fisher', 'SVM'} )} = 'Logisitic'
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@representationModel( superArgsCell{:}, ...
                                             ZDim = ZDim, ...
                                             XChannels = XChannels, ...
                                             NumCompLines = 8 );

            self.XDim = XDim;
            self.XOutputDim = XOutputDim;
            self.CDim = CDim;

            % placeholders for subclasses to define
            self.Nets.Encoder = [];
            self.Nets.Decoder = [];
            self.NetNames = {'Encoder', 'Decoder'};
            self.NumNetworks = 2;
            self.IsVAE = args.isVAE;
            self.NumVAEDraws = args.numVAEDraws;
            self.HasSeqInput = args.hasSeqInput;

            self.AuxModelType = args.auxModel;
            self.AuxModel = [];

            % copy over the loss functions associated
            % and any networks with them for later training 
            self = addLossFcns( self, lossFcns{:}, weights = args.weights );

            self.NumLoss = sum( self.LossFcnTbl.NumLosses );

            % indicate that further initialization is required
            self.IsInitialized = false;

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
            self.LossFcnWeights = [ self.LossFcnWeights w ];

            % update the names list
            self.LossFcnNames = [ self.LossFcnNames getFcnNames(newFcns) ];
            % add to the loss functions
            for i = 1:length( newFcns )
                self.LossFcns.(newFcns{i}.Name) = newFcns{i};
            end

            % add networks, if required
            self = addLossFcnNetworks( self, newFcns );
          
            % store the loss functions' details 
            % and relevant details for easier access when training
            self = self.setLossInfoTbl;
            self.LossFcnTbl.Types = categorical( self.LossFcnTbl.Types );
            self.LossFcnTbl.Inputs = categorical( self.LossFcnTbl.Inputs );

            % check a reconstruction loss is present
            if ~any( self.LossFcnTbl.Types=='Reconstruction' )
                eid = 'aeModel:NoReconstructionLoss';
                msg = 'No reconstruction loss object has been specified.';
                throwAsCaller( MException(eid,msg) );
            end 

            % check there is no more than one auxiliary network, if at all
            auxFcns = self.LossFcnTbl.Types=='Auxiliary';
            if sum( auxFcns ) > 1
                eid = 'aeModel:MulitpleAuxiliaryFunction';
                msg = 'There is more than one auxiliary loss function.';
                throwAsCaller( MException(eid,msg) );
            elseif sum( auxFcns ) == 1
                % for convenience identify the auxiliary network
                auxNet = (self.NetNames == self.LossFcnNames(auxFcns));
                self.AuxNetwork = self.NetNames( auxNet );
            end

        end


        function self = initTrainer( self, args )
            arguments
                self
                args.?modelTrainer
            end

            % set the trainer's properties
            argsCell = namedargs2cell( args );
            self.Trainer = modelTrainer( self.LossFcnTbl, ...
                                         argsCell{:}, ...
                                         showPlots = self.ShowPlots );

            % confirm if initialization is complete
            self.IsInitialized = ~isempty(self.Optimizer);

        end
        

        function self = initOptimizer( self, args )
            arguments
                self
                args.?modelOptimizer
            end

            % set the trainer's properties
            argsCell = namedargs2cell( args );
            self.Optimizer = modelOptimizer( self.NetNames, argsCell{:} );

            % confirm if initialization is complete
            self.IsInitialized = ~isempty(self.Trainer);

        end


        function self = train( self, thisDataset )
            % Train the autoencoder model
            arguments
                self          autoencoderModel
                thisDataset   modelDataset
            end

            if ~self.IsInitialized
                eid = 'Autoencoder:NotInitialized';
                msg = 'The trainer, optimizer or dataset parameters have not all been set.';
                throwAsCaller( MException(eid,msg) );
            end

            % check dataset is suitable
            if thisDataset.isFixedLength == self.HasSeqInput
                eid = 'aeModel:DatasetNotSuitable';
                if thisDataset.isFixedLength
                    msg = 'The dataset should have variable length for the model.';
                else
                    msg = 'The dataset should have fixed length for the model.';
                end
                throwAsCaller( MException(eid,msg) );
            end

            % run the training loop
            [ self, self.Optimizer ] = ...
                            self.Trainer.runTraining( self, ...
                                                      thisTrnSet, ...
                                                      thisValSet );


        end


        function loss = getReconLoss( self, X, XHat )
            % Calculate the reconstruction loss
            arguments
                self            autoencoderModel
                X     
                XHat  
            end
            
            name = self.LossFcnTbl.Names( self.LossFcnTbl.Types=='Reconstruction' );
            loss = self.LossFcns.(name).calcLoss( X, XHat );

        end


        function loss = getReconTemporalLoss( self, X, XHat )
            % Calculate the reconstruction loss over time
            arguments
                self            autoencoderModel
                X     
                XHat  
            end
            
            name = self.LossFcnTbl.Names( self.LossFcnTbl.Types=='Reconstruction' );
            loss = self.LossFcns.(name).calcTemporalLoss( X, XHat );

        end


        function setScalingFactor( self, data )
            % Set the scaling factors for reconstructions
            arguments
                self            autoencoderModel
                data            double
            end
            
            for i = 1:size( self.LossFcnTbl, 1 )
                
                if ismember( self.LossFcnTbl.Inputs(i), {'X-XHat', 'XC', 'XHat'} )
                    name = self.LossFcnTbl.Names(i);
                    self.LossFcns.(name).setScale( data );
                end
    
            end

        end


        function labels = get.XDimLabels( self )
            % Get the X dimensional labels for dlarrays
            arguments
                self            autoencoderModel
            end

            if self.XChannels==1
                if self.HasSeqInput
                    labels = 'TBC';
                else
                    labels = 'CB';
                end
            else
                if self.HasSeqInput
                    labels = 'TBC';
                else
                    labels = 'SBC';
                end
            end
            
        end


        function labels = get.XNDimLabels( self )
            % Get the XN dimensional labels for dlarrays
            arguments
                self            autoencoderModel
            end

            if self.XChannels==1
                labels = 'CB';
            else
                labels = 'SBC';
            end
            
        end


        function [ dlXC, offsets ] = latentComponents( self, dlZ, args )
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
                args.forward    logical = false
                args.convert    logical = false
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
            
            % initialise the components' Z codes at the mean
            % include an extra one that will be preserved
            dlZC = repmat( dlZMean, 1, ZDim*nSample+1 );

            % convert to double to speed up processing
            % provided dlZC can trace back to original dlZ
            Z = double(extractdata( dlZ ));
            
            % define the offset spacing, which must sum to zero
            switch args.sampling
                case 'Random'
                    % generate centred uniform distribution
                    offsets = 2*args.range*(rand( nSample, 1 )-0.5);

                case 'Fixed'
                    offsets = linspace( -args.range, args.range, nSample );
            end
            % convert the z-scores (offsets) to percentiles
            % giving a preponderance of values at the tails
            prc = 100*normcdf( offsets );

            for j = 1:nSample
                
                for i =1:ZDim
                    % vary ith component randomly about its mean
                    dlZC(i,(i-1)*nSample+j) = prctile( Z(i,:), prc(j) );
                end

            end

            % generate all the component curves using the decoder
            if args.forward
                dlXC = forward( self.Nets.Decoder, dlZC );
            else
                dlXC = predict( self.Nets.Decoder, dlZC );
            end

            XDim = size( dlXC );

            if args.centre
                % centre about the mean curve (last curve)
                if length( XDim )==2
                    dlXC = dlXC( :, 1:end-1 ) - dlXC( :, end );
                else
                    dlXC = dlXC( :, :, 1:end-1 ) - dlXC( :, :, end );
                end
            end

            if args.convert
                dlXC = double(extractdata( dlXC ));
            end

        end


        function [ dlXHat, dlZ, state ] = forward( self, encoder, decoder, dlX )
            % Forward-run the autoencoder networks
            arguments
                self        autoencoderModel
                encoder     dlnetwork
                decoder     dlnetwork
                dlX         dlarray
            end

            % generate latent encodings
            [ dlZ, state.Encoder ] = forward( encoder, dlX );
    
            % reconstruct curves from latent codes
            [ dlXHat, state.Decoder ] = forward( decoder, dlZ );

        end


        function dlZ = encode( self, X, arg )
            % Encode features Z from X using the model
            arguments
                self            autoencoderModel
                X
                arg.convert     logical = true
            end

            if isa( X, 'modelDataset' )
                dlX = X.getDLInput( self.XDimLabels );
            elseif isa( X, 'dlarray' )
                dlX = X;
            else
                eid = 'Autoencoder:NotValidX';
                msg = 'The input data should be a modelDataset or a dlarray.';
                throwAsCaller( MException(eid,msg) );
            end

            dlZ = predict( self.Nets.Encoder, dlX );

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

            dlXHat = predict( self.Nets.Decoder, dlZ );

            if arg.convert
                dlXHat = double(extractdata( dlXHat ));
                dlXHat = permute( dlXHat, [1 3 2] );
            end
            

        end


        function [ YHat, loss ] = predictAux( self, Z, Y )
            % Make prediction from Z using an auxiliary network
            arguments
                self            autoencoderModel
                Z
                Y
            end

            if isa( Z, 'dlarray' )
                dlZ = Z;
            else
                dlZ = dlarray( Z', 'CB' );
            end

            if isa( Y, 'dlarray' )
                dlY = Y;
            else
                dlY = dlarray( Y, 'CB' );
            end

            auxNet = (self.LossFcnTbl.Types == 'Auxiliary');
            if ~any( auxNet )
                eid = 'aeModel:NoAuxiliaryFunction';
                msg = 'No auxiliary loss function specified in the model.';
                throwAsCaller( MException(eid,msg) );
            end

            if ~self.LossFcnTbl.HasNetwork( auxNet )
                eid = 'aeModel:NoAuxiliaryNetwork';
                msg = 'No auxiliary network specified in the model.';
                throwAsCaller( MException(eid,msg) );
            end

            auxNetName = self.LossFcnTbl.Names( auxNet );

            dlYHat = predict( self.Nets.(auxNetName), dlZ );

            loss = calcLoss( self.LossFcns.(auxNetName), ...
                             self.Nets.(auxNetName), ...
                             dlZ, dlY );

            YHat = double(extractdata( dlYHat ))';
            loss = double(extractdata( loss ));

        end


        function [ YHat, loss ] = predictComparator( self, X, Y )
            % Make prediction from X using the comparator network
            arguments
                self            autoencoderModel
                X
                Y
            end

            if isa( X, 'dlarray' )
                dlX = X;
            else
                dlX = dlarray( X', 'CB' );
            end

            if isa( Y, 'dlarray' )
                dlY = Y;
            else
                dlY = dlarray( Y, 'CB' );
            end

            compNet = (self.LossFcnTbl.Types == 'Comparator');
            if ~any( compNet )
                eid = 'aeModel:NoComparatorFunction';
                msg = 'No comparator loss function specified in the model.';
                throwAsCaller( MException(eid,msg) );
            end

            if ~self.LossFcnTbl.HasNetwork( compNet )
                eid = 'aeModel:NoComparatorNetwork';
                msg = 'No comparator network specified in the model.';
                throwAsCaller( MException(eid,msg) );
            end

            compNetName = self.LossFcnTbl.Names( compNet );

            dlYHat = predict( self.Nets.(compNetName), dlX );

            loss = calcLoss( self.LossFcns.(compNetName), ...
                             self.Nets.(compNetName), ...
                             dlX, dlY );

            YHat = double(extractdata( dlYHat ))';
            loss = double(extractdata( loss ));

        end



        function net = getNetwork( self, name )
            arguments
                self
                name         string {mustBeNetName( self, name )}
            end
            
            net = self.Nets.(name);
            if isa( net, 'lossFunction' )
                net = self.LossFcns.(name).net;
            end

        end

        
        function names = getNetworkNames( self )
            arguments
                self
            end
            
            names = fieldnames( self.Nets );

        end


        function isValid = mustBeNetName( self, name )
            arguments
                self
                name
            end

            isValid = ismember( name, self.names );

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



    methods (Access = protected)


        function self = addLossFcnNetworks( self, newFcns )
            % Add one or more networks to the model
            arguments
                self        autoencoderModel
                newFcns     cell
            end

            nFcns = length( newFcns );
            k = length( self.Nets );
            for i = 1:nFcns
                thisLossFcn = newFcns{i};
                if thisLossFcn.HasNetwork
                    k = k+1;
                    % set the data dimensions 
                    thisLossFcn = setDimensions( thisLossFcn, self );
                    % add the network object
                    self.Nets.(thisLossFcn.Name) = initNetwork( thisLossFcn );
                    % record its name
                    self.NetNames = [ string(self.NetNames) thisLossFcn.Name ];
                    % increment the counter
                    self.NumNetworks = self.NumNetworks + 1;
                end
            end


        end

        function self = setLossInfoTbl( self )
            % Update the info table
            
            nFcns = length( self.LossFcnNames );
            Names = strings( nFcns, 1 );
            Types = strings( nFcns, 1 );
            Inputs = strings( nFcns, 1 );
            Weights = zeros( nFcns, 1 );
            NumLosses = zeros( nFcns, 1 );
            LossNets = strings( nFcns, 1 );
            HasNetwork = false( nFcns, 1 );
            DoCalcLoss = false( nFcns, 1 );
            UseLoss = false( nFcns, 1 );

            for i = 1:nFcns
                
                thisLossFcn = self.LossFcns.(self.LossFcnNames(i));
                Names(i) = thisLossFcn.Name;
                Types(i) = thisLossFcn.Type;
                Inputs(i) = thisLossFcn.Input;
                Weights(i) = self.LossFcnWeights(i);
                NumLosses(i) = thisLossFcn.NumLoss;
                HasNetwork(i) = thisLossFcn.HasNetwork;
                DoCalcLoss(i) = thisLossFcn.DoCalcLoss;
                UseLoss(i) = thisLossFcn.UseLoss;

                nFcnNets = length( thisLossFcn.LossNets );
                for j = 1:nFcnNets
                    if length(string( thisLossFcn.LossNets{j} ))==1
                        assignments = thisLossFcn.LossNets{j};
                    else
                        assignments = strjoin( thisLossFcn.LossNets{j,:}, '+' );
                    end
                    LossNets(i) = strcat( LossNets(i), assignments ) ;
                    if j < nFcnNets
                        LossNets(i) = strcat( LossNets(i), "; " );
                    end
                end

            end

            self.LossFcnTbl = table( Names, Types, Inputs, Weights, ...
                    NumLosses, LossNets, HasNetwork, DoCalcLoss, UseLoss );

        end

    end


end


function names = getFcnNames( lossFcns )

    nFcns = length( lossFcns );
    names = strings( nFcns, 1 );
    for i = 1:nFcns
        names(i) = lossFcns{i}.Name;
    end

end



