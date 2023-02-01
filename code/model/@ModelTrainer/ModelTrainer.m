classdef ModelTrainer < handle
    % Class defining a model trainer

    properties
        NumIterations    % maximum number of iterations for training
        NumIterPreTrn    % number of iterations for pretraining
        CurrentIteration % iteration counter
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

        ShowPlots        % flag whether to show plots
        LossFig          % figure for the loss lines
        LossLines        % animated lines cell array
    end

    methods

        function self = ModelTrainer( lossFcnTbl, args )
            % Initialize the model
            arguments
                lossFcnTbl          table
                args.NumIterations  double ...
                    {mustBeInteger, mustBePositive} = 1000;
                args.NumIterPreTrn  double ...
                    {mustBeInteger, ...
                     mustBeGreaterThanOrEqual(args.NumIterPreTrn,0) } = 0;
                args.BatchSize      double ...
                    {mustBeInteger, mustBePositive} = 100;
                args.PartialBatch   char ...
                    {mustBeMember(args.PartialBatch, ...
                        {'discard', 'return'} )} = 'return'
                args.HasMiniBatchShuffle logical = true
                args.HasShuffleRandomStream  logical = false
                args.Holdout        double ...
                    {mustBeInRange( args.Holdout, 0, 0.5 )} = 0.2
                args.ValFreq        double ...
                    {mustBeInteger, mustBePositive} = 50; 
                args.UpdateFreq     double ...
                    {mustBeInteger, mustBePositive} = 50;
                args.LRFreq         double ...
                    {mustBeInteger, mustBePositive} = 500;
                args.ValPatience    double ...
                    {mustBeInteger, mustBePositive} = 20;
                args.ActiveZFreq    double ...
                    {mustBeInteger, mustBePositive} = 50;
                args.PostTraining   logical = true;
                args.ValType        char ...
                    {mustBeMember(args.ValType, ...
                        {'Reconstruction', 'AuxNetwork', 'Both'} )} ...
                            = 'Both'
                args.ShowPlots      logical = false

            end

            % initialize the training parameters
            self.NumIterations = args.NumIterations;
            self.NumIterPreTrn = args.NumIterPreTrn;
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