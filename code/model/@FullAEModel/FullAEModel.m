classdef FullAEModel < FullRepresentationModel
    % Subclass defining the framework for an autoencoder model
    
    properties
        NetNames       % names of the networks (for convenience)
        NumNetworks    % number of networks
        IdenticalNetInit % wherther to use same initialized networks
        InitializedNets% initialized networks before training
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
        ComponentCentring % how to centre the generated components
        HasCentredDecoder % whether the decoder predicts centred X
    end

    properties (Dependent = true)
        XDimLabels     % dimensional labelling for X input dlarrays
        XNDimLabels    % dimensional labelling for time-normalized output
    end

    methods

        function self = FullAEModel( thisDataset, ...
                                     superArgs, ...
                                     superArgs2, ...
                                     args )
            % Initialize the model
            arguments
                thisDataset         ModelDataset
            end
            arguments
                superArgs.?FullRepresentationModel
                superArgs2.name     string
                superArgs2.path     string
                args.IdenticalNetInit logical = false
                args.FlattenInput   logical = false
                args.HasSeqInput    logical = false
                args.InitZDimActive double ...
                    {mustBeInteger} = 1
                args.ComponentCentring string ...
                    {mustBeMember( args.ComponentCentring, ...
                                    {'Z', 'X', 'None'} )} = 'Z'
                args.HasCentredDecoder logical = true
                args.Weights        double ...
                                    {mustBeNumeric,mustBeVector} = 1
                args.Trainer        struct = []
                args.Optimizer      struct = []
                args.LossFcns       struct = []
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
            self.FlattenInput = args.FlattenInput;
            self.HasSeqInput = args.HasSeqInput;
            self.ComponentCentring = args.ComponentCentring;
            self.HasCentredDecoder = args.HasCentredDecoder;

            if args.InitZDimActive==0
                self.InitZDimActive = self.ZDim;
            else
                self.InitZDimActive = min( args.InitZDimActive, self.ZDim );
            end

            self.Trainer = args.Trainer;
            self.Optimizer = args.Optimizer;

            % copy over the loss functions associated
            % and any networks with them for later training
            
            self = self.addLossFcns( lossFcns{:}, weights = args.Weights );

            self.NumLoss = sum( self.LossFcnTbl.NumLosses );

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


        % class methods

        self = addLossFcnNetworks( self )

        self = addLossFcns( self, newFcns, args )

        self = computeCVParameters( self )

        self = conserveMemory( self, level )
        
        self = initSubModel( self, k )

        self = setLossScalingFactor( self )

        self = plotAllALE( self )

        [ YHatFold, YHatMaj ] = predictAuxNet( self, Z, Y )

        [ YHatFold, YHatMaj ] = predictCompNet( self, thisDataset )

        self = setLossInfoTbl( self )

    end

end