classdef ModelTrainer < handle
    % Class defining a model trainer

    properties
        NumEpochs        % maximum number of epochs for training
        NumEpochsPreTrn  % number of epochs for pretraining
        CurrentEpoch     % epoch counter
        BatchSize        % minibatch size
        PartialBatch     % what to do with an incomplete batch
        HasMiniBatchShuffle % indicates if minibatches are shuffled
        HasShuffleRandomStream % indicates if separate random stream for minibatch shuffling

        Holdout          % proportion of the dataset for validation
        ValFreq          % validation frequency in epochs
        UpdateFreq       % update frequency in epochs
        LRFreq           % learning rate update frequency
        ActiveZFreq      % active Z dimensions update frequency

        ValPatience      % validation patience in valFreq units
        ValType          % validation function name

        NumLossFcns      % number of loss functions
        LossTrn          % record of training losses
        LossVal          % record of validation losses

        Metrics          % record of training metrics
        
        PreTraining      % flag to indicate AE training
        PostTraining     % flag to indicate whether to continue training

        ShowPlots        % flag whether to show plots
        LossFig          % figure for the loss lines
        LossLines        % animated lines cell array
    end

    methods

        function self = ModelTrainer( lossFcnTbl, args )
            % Initialize the model
            arguments
                lossFcnTbl          table
                args.NumEpochs      double ...
                    {mustBeInteger, mustBePositive} = 2000;
                args.NumEpochsPreTrn  double ...
                    {mustBeInteger, ...
                     mustBeGreaterThanOrEqual(args.NumEpochsPreTrn,0) } = 0;
                args.BatchSize      double ...
                    {mustBeInteger, mustBePositive} = 40;
                args.PartialBatch   char ...
                    {mustBeMember(args.PartialBatch, ...
                        {'discard', 'return'} )} = 'discard'
                args.HasMiniBatchShuffle logical = true
                args.HasShuffleRandomStream  logical = false
                args.Holdout        double ...
                    {mustBeInRange( args.Holdout, 0, 0.5 )} = 0.2
                args.ValFreq        double ...
                    {mustBeInteger, mustBePositive} = 1; 
                args.UpdateFreq     double ...
                    {mustBeInteger, mustBePositive} = 50;
                args.LRFreq         double ...
                    {mustBeInteger, mustBePositive} = 100;
                args.ValPatience    double ...
                    {mustBeInteger, mustBePositive} = 100;
                args.ActiveZFreq    double ...
                    {mustBeInteger, mustBePositive} = 25;
                args.PostTraining   logical = true;
                args.ValType        char ...
                    {mustBeMember(args.ValType, ...
                        {'Reconstruction', 'AuxNetwork', 'Both'} )} ...
                            = 'Both'
                args.ShowPlots      logical = false

            end

            % initialize the training parameters
            self.NumEpochs = args.NumEpochs;
            self.NumEpochsPreTrn = args.NumEpochsPreTrn;
            self.CurrentEpoch = 0;
            self.BatchSize = args.BatchSize;
            self.PartialBatch = args.PartialBatch;

            self.Holdout = args.Holdout;
            self.ValFreq = args.ValFreq;
            self.UpdateFreq = args.UpdateFreq;
            self.LRFreq = args.LRFreq;
            self.ActiveZFreq = args.ActiveZFreq;

            self.ValPatience = args.ValPatience;
            self.ValType = args.ValType;

            self.HasMiniBatchShuffle = args.HasMiniBatchShuffle;
            self.HasShuffleRandomStream = args.HasShuffleRandomStream;

            self.NumLossFcns = size( lossFcnTbl,1 );
            self.LossTrn = [];
            self.LossVal = [];

            self.PreTraining = true;
            self.PostTraining = args.PostTraining;

            self.ShowPlots = args.ShowPlots;

            if self.ShowPlots
                [self.LossFig, self.LossLines] = ...
                                initializeLossPlots( lossFcnTbl );
            end

        end

        % class methods

        thisModel = runTraining( self, thisModel, thisDataset );

        thisModel = runTrainingLoop( self, ...
                                     thisModel, mbqTrn, ...
                                     validationFcn, ...
                                     metricsFcn, ...
                                     reportFcn, ...
                                     isFixedLength )

    end

    methods (Static, Access= protected)

        [grad, state, loss] = gradients( nets, ...
                                         thisModel, ...
                                         dlXIn, ...
                                         dlXOut, ...
                                         dlP, ...
                                         dlY, ...
                                         preTraining )
        
        i = iterationsPerEpoch( mbq )

    end

end