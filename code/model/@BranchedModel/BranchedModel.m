classdef BranchedModel < AEModel
    % A branched autoencoder
    % Subclasses define the hidden layers of the encoder and decoder networks
    properties
        HasInputNormalization % whether to apply amplitude normalization
        NumHidden             % number of hidden layers in the encoder
        NumHiddenDecoder      % number of hidden layers in the decoder
        NetNormalizationType  % normalization for the encoder
        NetNormalizationTypeDecoder  % normalization for the decoder

        NetActivationType     % nonlinear activation function for the encoder
        NetActivationTypeDecoder     % nonlinear activation function for the decoder
        ReluScale             % leaky ReLu scale factor for the encoder
        ReluScaleDecoder      % leaky ReLu scale factor for the decoder
        Dropout               % hidden layer dropout rate for the encoder
        DropoutDecoder        % hidden layer dropout rate for the decoder 
        InputDropout          % input dropout rate
        HasBranchedEncoder    % whether to have branching in the encoder
        HasEncoderMasking     % whether to mask Z outputs from the encoder
        HasBranchedDecoder    % whether to have branching in the decoder
        HasDecoderMasking     % whether to mask Z inputs to the decoder
    end

    methods

        function self = BranchedModel( thisDataset, ...
                                           superArgs, ...
                                           superArgs2, ...
                                           args )
            % Initialize the model
            arguments
                thisDataset                 ModelDataset
                superArgs.?AEModel
                superArgs2.name             string
                superArgs2.path             string
                args.HasInputNormalization  logical = true
                args.NumHidden              double ...
                    {mustBeInteger, mustBePositive} = 2
                args.NumHiddenDecoder       double ...
                    {mustBeInteger, mustBePositive} = 2
                args.NetNormalizationType char ...
                    {mustBeMember( args.NetNormalizationType, ...
                    {'None', 'Batch', 'Layer'} )} = 'None'
                args.NetNormalizationTypeDecoder char ...
                    {mustBeMember( args.NetNormalizationTypeDecoder, ...
                    {'None', 'Batch', 'Layer'} )} = 'None'
                args.NetActivationType char ...
                    {mustBeMember( args.NetActivationType, ...
                    {'None', 'Tanh', 'Relu'} )} = 'Tanh'
                args.NetActivationTypeDecoder char ...
                    {mustBeMember( args.NetActivationTypeDecoder, ...
                    {'None', 'Tanh', 'Relu'} )} = 'Tanh'
                args.ReluScale      double ...
                    {mustBeInRange(args.ReluScale, 0, 1)} = 0.2
                args.ReluScaleDecoder       double ...
                    {mustBeInRange(args.ReluScaleDecoder, 0, 1)} = 0.2
                args.InputDropout   double ...
                    {mustBeInRange(args.InputDropout, 0, 1)} = 0.2
                args.Dropout                double ...
                    {mustBeInRange(args.Dropout, 0, 1)} = 0.05
                args.DropoutDecoder         double ...
                    {mustBeInRange(args.DropoutDecoder, 0, 1)} = 0.0
                args.HasBranchedEncoder     logical = false
                args.HasBranchedDecoder     logical = true
                args.HasEncoderMasking      logical = false
                args.HasDecoderMasking      logical = true
                args.FlattenInput           logical = true
                args.HasSeqInput            logical = false
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            superArgs2Cell = namedargs2cell( superArgs2 );

            self@AEModel( thisDataset, ...
                          superArgsCell{:}, ...
                          superArgs2Cell{:}, ...
                          FlattenInput = args.FlattenInput, ...
                          HasSeqInput = args.HasSeqInput );

            % store this class's properties
            self.HasInputNormalization = args.HasInputNormalization;
            self.NumHidden = args.NumHidden;
            self.NumHiddenDecoder = args.NumHiddenDecoder;
            self.NetNormalizationType = args.NetNormalizationType;
            self.NetNormalizationTypeDecoder = args.NetNormalizationTypeDecoder;
            self.NetActivationType = args.NetActivationType;
            self.NetActivationTypeDecoder = args.NetActivationTypeDecoder;
            self.ReluScale = args.ReluScale;
            self.ReluScaleDecoder = args.ReluScaleDecoder;
            self.Dropout = args.Dropout;
            self.DropoutDecoder = args.DropoutDecoder;
            self.InputDropout = args.InputDropout;
            self.HasBranchedEncoder = args.HasBranchedEncoder;
            self.HasEncoderMasking = args.HasEncoderMasking;
            self.HasBranchedDecoder = args.HasBranchedDecoder;
            self.HasDecoderMasking = args.HasDecoderMasking;
           
        end

    end


    methods (Abstract)

        % abstract methods to be defined by subclasses

        initEncoderInputLayers;

        initEncoderHiddenLayers;
        
        initDecoderInputLayers;

        initDecoderHiddenLayers;
      
    end


    methods (Static)

        [ lgraph, lastLayer ] = addBlock( firstLayer, ...
                                lgraph, i, precedingLayer,  ...
                                scale, dropout, normType, actType )

    end
    

end
