classdef FullAEModel < FullRepresentationModel
    % Subclass defining the framework for an autoencoder model
    
    properties
        NetNames       % names of the networks (for convenience)
        AuxNetName     % name of the auxiliary network
        NumNetworks    % number of networks
        IsVAE          % flag indicating if variational autoencoder
        NumVAEDraws    % number of draws from encoder output distribution
        LossFcns       % loss function objects
        LossFcnNames   % names of the loss functions
        LossFcnWeights % weights to be applied to the loss function
        LossFcnTbl     % convenient table summarising loss function details
        NumLoss        % number of computed losses
        FlattenInput   % whether to flatten inputs
        HasSeqInput    % supports variable-length input
        Trainer        % optional arguments for the trainer
        Optimizer      % optional arguments for the optimizer
    end

    properties (Dependent = true)
        XDimLabels     % dimensional labelling for X input dlarrays
        XNDimLabels    % dimensional labelling for time-normalized output
    end

    methods

        function self = FullAEModel( thisDataset, ...
                                     lossFcns, ...
                                     superArgs, ...
                                     superArgs2, ...
                                     args )
            % Initialize the model
            arguments
                thisDataset         ModelDataset
            end
            arguments (Repeating)
                lossFcns            LossFunction
            end
            arguments
                superArgs.?FullRepresentationModel
                superArgs2.name     string
                superArgs2.path     string
                args.isVAE          logical = false
                args.numVAEDraws    double ...
                    {mustBeInteger, mustBePositive} = 1
                args.flattenInput   logical = false
                args.hasSeqInput    logical = false
                args.weights        double ...
                                    {mustBeNumeric,mustBeVector} = 1
                args.trainer        struct = []
                args.optimizer      struct = []
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            superArgs2Cell = namedargs2cell( superArgs2 );
            self = self@FullRepresentationModel( thisDataset, ...
                                                 superArgsCell{:}, ...
                                                 superArgs2Cell{:}, ...
                                                 NumCompLines = 9 );

            % check dataset is suitable
            if thisDataset.isFixedLength == args.hasSeqInput
                eid = 'FullAEModel:DatasetNotSuitable';
                if thisDataset.isFixedLength
                    msg = 'The dataset should have variable length for the model.';
                else
                    msg = 'The dataset should have fixed length for the model.';
                end
                throwAsCaller( MException(eid,msg) );
            end

            % placeholders for subclasses to define
            self.NetNames = {'Encoder', 'Decoder'};
            self.NumNetworks = 2;
            self.IsVAE = args.isVAE;
            self.NumVAEDraws = args.numVAEDraws;
            self.FlattenInput = args.flattenInput;
            self.HasSeqInput = args.hasSeqInput;

            self.Trainer = args.trainer;
            self.Optimizer = args.optimizer;

            % copy over the loss functions associated
            % and any networks with them for later training 
            self = addLossFcns( self, lossFcns{:}, weights = args.weights );

            self.NumLoss = sum( self.LossFcnTbl.NumLosses );

        end


        function self = addLossFcns( self, newFcns, args )
            % Add one or more loss function objects to the model
            arguments
                self
            end
            arguments (Repeating)
                newFcns   LossFunction
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

            % add details associated with the loss function networks
            % but without initializing them
            self = addLossFcnNetworks( self, newFcns );

            % store the loss functions' details 
            % and relevant details for easier access when training
            self = self.setLossInfoTbl;
            self.LossFcnTbl.Types = categorical( self.LossFcnTbl.Types );
            self.LossFcnTbl.Inputs = categorical( self.LossFcnTbl.Inputs );

            % set loss function scaling factors if required
            self = self.setLossScalingFactor;

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
                self.AuxNetName = self.NetNames( auxNet );
            end

        end


        function thisModel = initSubModel( self, id )
            % Initialize a sub-model
            arguments
                self            FullAEModel
                id              double
            end

            thisModel = CompactAEModel( self, id );

        end


        function self = setLossScalingFactor( self )
            % Set the scaling factors for reconstructions
            arguments
                self            FullAEModel
            end
            
            for i = 1:size( self.LossFcnTbl, 1 )
                
                if ismember( self.LossFcnTbl.Inputs(i), {'X-XHat', 'XC', 'XHat'} )
                    name = self.LossFcnTbl.Names(i);
                    self.LossFcns.(name).Scale = self.Scale;
                end
    
            end

        end


        function labels = get.XDimLabels( self )
            % Get the X dimensional labels for dlarrays
            arguments
                self            FullAEModel
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
                self            FullAEModel
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
                self            FullAEModel
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


        function [ YHatFold, YHatMaj ] = predictAuxNet( self, Z, Y )
            % Predict Y from Z using all trained auxiliary networks
            arguments
                self            FullAEModel
                Z               {mustBeA(Z, {'double', 'dlarray'})}
                Y               {mustBeA(Y, {'double', 'dlarray'})}
            end

            isEnsemble = (size( Z, 3 ) > 1);
            nRows = size( Z, 1 );
            YHatFold = zeros( nRows, self.KFolds );
            for k = 1:self.KFolds
                if isEnsemble
                    YHatFold( :, k ) = ...
                            predictAuxNet( self.SubModels{k}, Z(:,:,k), Y );
                else
                    YHatFold( :, k ) = ...
                            predictAuxNet( self.SubModels{k}, Z, Y );
                end
            end

            YHatMaj = zeros( nRows, 1 );
            for i = 1:nRows
                [votes, grps] = groupcounts( YHatFold(i,:)' );
                [ ~, idx ] = max( votes );
                YHatMaj(i) = grps( idx );
            end

        end


        function [ YHatFold, YHatMaj ] = predictCompNet( self, thisDataset )
            % Predict Y from X using all comparator networks
            arguments
                self            FullAEModel
                thisDataset     ModelDataset
            end

            YHatFold = zeros( thisDataset.NumObs, self.KFolds );
            for k = 1:self.KFolds
                YHatFold( :, k ) = predictCompNet( self.SubModels{k}, thisDataset );
            end

            YHatMaj = zeros( thisDataset.NumObs, 1 );
            for i = 1:thisDataset.NumObs
                [votes, grps] = groupcounts( YHatFold(i,:)' );
                [ ~, idx ] = max( votes );
                YHatMaj(i) = grps( idx );
            end

        end


    end



    methods (Access = protected)


        function self = addLossFcnNetworks( self, newFcns )
            % Add one or more networks to the model
            arguments
                self        FullAEModel
                newFcns     cell
            end

            nFcns = length( newFcns );
            for i = 1:nFcns
                thisLossFcn = newFcns{i};
                if thisLossFcn.HasNetwork
                    % set the data dimensions 
                    thisLossFcn = setDimensions( thisLossFcn, self );
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



