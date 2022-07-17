classdef FullAEModel < FullRepresentationModel
    % Subclass defining the framework for an autoencoder model
    
    properties
        NetNames       % names of the networks (for convenience)
        NumNetworks    % number of networks
        IdenticalNetInit % wherther to use same initialized networks
        InitializedNets% initialized networks before training
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
        InitZDimActive % initial number of Z dimensions active
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
                args.IdenticalNetInit logical = false
                args.IsVAE          logical = false
                args.NumVAEDraws    double ...
                    {mustBeInteger, mustBePositive} = 1
                args.FlattenInput   logical = false
                args.HasSeqInput    logical = false
                args.InitZDimActive double ...
                    {mustBeInteger} = 1
                args.Weights        double ...
                                    {mustBeNumeric,mustBeVector} = 1
                args.Trainer        struct = []
                args.Optimizer      struct = []
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            superArgs2Cell = namedargs2cell( superArgs2 );
            self = self@FullRepresentationModel( thisDataset, ...
                                                 superArgsCell{:}, ...
                                                 superArgs2Cell{:}, ...
                                                 NumCompLines = 5 );

            % check dataset is suitable
            if thisDataset.isFixedLength == args.HasSeqInput
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
            self.IdenticalNetInit = args.IdenticalNetInit; 
            self.IsVAE = args.IsVAE;
            self.NumVAEDraws = args.NumVAEDraws;
            self.FlattenInput = args.FlattenInput;
            self.HasSeqInput = args.HasSeqInput;

            if args.InitZDimActive==0
                self.InitZDimActive = self.ZDim;
            else
                self.InitZDimActive = min( args.InitZDimActive, self.ZDim );
            end

            self.Trainer = args.Trainer;
            self.Optimizer = args.Optimizer;

            % copy over the loss functions associated
            % and any networks with them for later training 
            self = addLossFcns( self, lossFcns{:}, weights = args.Weights );

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
                eid = 'FullAEModel:MultipleAuxiliaryFunction';
                msg = 'There is more than one auxiliary loss function.';
                throwAsCaller( MException(eid,msg) );
            end

            % identify the comparator network, if present
            comparatorFcns = self.LossFcnTbl.Types=='Comparator';
            if sum( comparatorFcns ) > 1
                eid = 'FullAEModel:MultipleComparatorFunction';
                msg = 'There is more than one comparator loss function.';
                throwAsCaller( MException(eid,msg) );
            end

        end


        function self = initSubModel( self, k )
            % Initialize a sub-model
            arguments
                self            FullAEModel
                k               double
            end

            self.SubModels{k} = CompactAEModel( self, k );
            if self.IdenticalNetInit && k==1
                self.InitializedNets = self.SubModels{k}.Nets;
            end

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


        function self = conserveMemory( self, level )
            % Conserve memory usage for AE
            arguments
                self            FullRepresentationModel
                level           double {mustBeInteger, mustBePositive} = 0
            end

            self = conserveMemory@FullRepresentationModel( self, level );

            if level >= 3
                for k = 1:self.KFolds
                    self.SubModels{k}.Optimizer = [];
                end
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



