classdef CompactAEModel < CompactRepresentationModel
    % Subclass defining the framework for an autoencoder model
    
    properties
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
        FlattenInput   % whether to flatten input
        HasSeqInput    % supports variable-length input
        AuxNetwork     % name of auxiliary dlnetwork

        Trainer        % trainer object holding training parameters
        Optimizer      % optimizer object
    end

    properties (Dependent = true)
        XDimLabels     % dimensional labelling for X input dlarrays
        XNDimLabels    % dimensional labelling for time-normalized output
    end

    methods

        function self = CompactAEModel( theFullModel, fold )
            % Initialize the model
            arguments
                theFullModel        FullAEModel
                fold                double
            end

            self@CompactRepresentationModel( theFullModel, fold );

            % copy over the full model's relevant properties
            self.NetNames = theFullModel.NetNames;
            self.NumNetworks = theFullModel.NumNetworks;
            self.IsVAE = theFullModel.IsVAE;
            self.NumVAEDraws = theFullModel.NumVAEDraws;
            self.LossFcns = theFullModel.LossFcns;
            self.LossFcnNames = theFullModel.LossFcnNames;
            self.LossFcnWeights = theFullModel.LossFcnWeights;
            self.LossFcnTbl = theFullModel.LossFcnTbl;
            self.NumLoss = theFullModel.NumLoss;
            self.FlattenInput = theFullModel.FlattenInput;
            self.HasSeqInput = theFullModel.HasSeqInput;

            % initialize the encoder and decoder networks
            self.Nets.Encoder = theFullModel.initEncoder;
            self.Nets.Decoder = theFullModel.initDecoder;

            % initialize the loss function networks, if required
            self = self.initLossFcnNetworks;

            % initialize the trainer
            try
                argsCell = namedargs2cell( theFullModel.Trainer );
            catch
                argsCell = {};
            end
            self.Trainer = ModelTrainer( self.LossFcnTbl, ...
                                         argsCell{:}, ...
                                         showPlots = self.ShowPlots );

            % initialize the optimizer
            try
                argsCell = namedargs2cell( theFullModel.Optimizer );
            catch
                argsCell = {};
            end
            self.Optimizer = ModelOptimizer( self.NetNames, argsCell{:} );

        end


        function self = train( self, thisTrnData, thisValData )
            % Train the autoencoder
            arguments
                self            CompactAEModel
                thisTrnData     ModelDataset
                thisValData     ModelDataset
            end

            self = self.Trainer.runTraining( self, thisTrnData, thisValData );

            % compute the components' explained variance
            [self.LatentComponents, self.VarProportion, self.ComponentVar] ...
                            = self.getLatentComponents( self, thisTrnData );

        end


        function labels = get.XDimLabels( self )
            % Get the X dimensional labels for dlarrays
            arguments
                self            CompactAEModel
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
                self            CompactAEModel
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
                self            CompactAEModel
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
                self        CompactAEModel
                encoder     dlnetwork
                decoder     dlnetwork
                dlX         dlarray
            end

            if self.FlattenInput && size( dlX, 3 ) > 1
                dlX = flattenDLArray( dlX );
            end

            % generate latent encodings
            [ dlZ, state.Encoder ] = forward( encoder, dlX );
    
            % reconstruct curves from latent codes
            [ dlXHat, state.Decoder ] = forward( decoder, dlZ );

        end


        function dlZ = encode( self, X, arg )
            % Encode features Z from X using the model
            arguments
                self            CompactAEModel
                X               {mustBeA(X, {'ModelDataset', 'dlarray'})}
                arg.convert     logical = true
            end

            if isa( X, 'ModelDataset' )
                dlX = X.getDLInput( self.XDimLabels );
            else
                dlX = X;
            end

            if self.FlattenInput && size( dlX, 3 ) > 1
                dlX = flattenDLArray( dlX );
            end

            dlZ = predict( self.Nets.Encoder, dlX );

            if arg.convert
                dlZ = double(extractdata( dlZ ))';
            end

        end


        function dlXHat = reconstruct( self, Z, arg )
            % Reconstruct X from Z using the model
            arguments
                self            CompactAEModel
                Z               {mustBeA(Z, {'double', 'dlarray'})}
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


        function [ YHat, loss ] = predictAuxNet( self, Z, Y )
            % Make prediction from Z using an auxiliary network
            arguments
                self            CompactAEModel
                Z               {mustBeA(Z, {'double', 'dlarray'})}
                Y               {mustBeA(Y, {'double', 'dlarray'})}
            end

            if isa( Z, 'dlarray' )
                dlZ = Z;
            else
                dlZ = dlarray( Z', 'CB' );
            end

            if isa( Y, 'dlarray' )
                Y = double(extractdata( Y ))';
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

            YHat = double(extractdata( dlYHat ))';
            CDim = size( YHat, 2 );
            YHat = double(onehotdecode( YHat, 1:CDim, CDim ));

            loss = getPropCorrect( Y, YHat );

        end


        function [ YHat, loss ] = predictCompNet( self, thisDataset )
            % Make prediction from X using the comparator network
            arguments
                self            CompactAEModel
                thisDataset     ModelDataset
            end

            dlX = thisDataset.getDLInput( self.XDimLabels );

            if self.FlattenInput && size( dlX, 3 ) > 1
                dlX = flattenDLArray( dlX );
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

            YHat = double(extractdata( dlYHat ))';
            YHat = double(onehotdecode( YHat, ...
                                1:thisDataset.CDim, thisDataset.CDim ));

            loss = getPropCorrect( thisDataset.Y, YHat );

        end


        function save( self )
            % Save the model plots and the object itself
            arguments
                self            CompactAEModel
            end

            plotObjects = self.Axes;
            plotObjects.Components = self.Figs.Components;
            plotObjects.Loss = self.Trainer.LossFig;

            savePlots( plotObjects, self.Info.Path, self.Info.Name );

        end


        function self = clearGraphics( self )
            % Clear the graphics objects to save memory
            % Including the loss lines figure
            arguments
                self            CompactAEModel
            end

            self = clearGraphics@CompactRepresentationModel( self );
            self.Trainer.LossFig = [];
            self.Trainer.LossLines = [];

        end

    end


    methods (Access = protected)


        function self = initLossFcnNetworks( self )
            % Initialize the loss function networks
            arguments
                self        CompactAEModel
            end

            for i = 1:length( self.LossFcnNames )
                thisName = self.LossFcnNames{i};
                thisLossFcn = self.LossFcns.(thisName);
                if thisLossFcn.HasNetwork
                    self.Nets.(thisName) = thisLossFcn.initNetwork;
                end
            end

        end


    end


    methods (Static)

        function [ eval, pred ] = evaluateSet( self, thisDataset )
            % Evaluate the model with a specified dataset
            % doing additional work to the superclass method
            arguments
                self             CompactAEModel
                thisDataset      ModelDataset
            end

            % call the superclass method
            [ eval, pred ] = ...
                evaluateSet@CompactRepresentationModel( self, thisDataset );

            if any(self.LossFcnTbl.Types == 'Comparator')
                % compute the comparator loss using the comparator network
                [ pred.ComparatorYHat, eval.ComparatorLoss ] = ...
                                predictCompNet( self, thisDataset ); 
            end
    
            if any(self.LossFcnTbl.Types == 'Auxiliary')
                % compute the auxiliary loss using the network
                [ pred.AuxNetworkYHat, eval.AuxNetworkLoss ] = ...
                                predictAuxNet( self, pred.Z, thisDataset.Y );
            end
        
        end

    end


end




