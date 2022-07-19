classdef ModelTrainer
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
                     mustBeGreaterThanOrEqual(args.NumEpochsPreTrn,0) } = 100;
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
                    {mustBeInteger, mustBePositive} = 5; 
                args.UpdateFreq     double ...
                    {mustBeInteger, mustBePositive} = 50;
                args.LRFreq         double ...
                    {mustBeInteger, mustBePositive} = 200;
                args.ValPatience    double ...
                    {mustBeInteger, mustBePositive} = 10;
                args.ActiveZFreq    double ...
                    {mustBeInteger, mustBePositive} = 25;
                args.PostTraining   logical = true;
                args.ValType        char ...
                    {mustBeMember(args.ValType, ...
                        {'Reconstruction', 'AuxNetwork', 'AuxModel'} )} ...
                            = 'Reconstruction'
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

        
        function thisModel = runTraining( self, thisModel, thisDataset )
            % Run the training loop for the model
            arguments
                self            ModelTrainer
                thisModel       CompactAEModel
                thisDataset     ModelDataset
            end

            % re-partition the data to create training and validation sets
            trainObs = thisDataset.getCVPartition( Holdout = self.Holdout );
            
            thisTrnData = thisDataset.partition( trainObs );
            thisValData = thisDataset.partition( ~trainObs );

            % setup the minibatch queues
            mbqTrn = thisTrnData.getMiniBatchQueue( ...
                                        self.BatchSize, ...
                                        thisModel.XDimLabels, ...
                                        thisModel.XNDimLabels, ...
                                        partialBatch = self.PartialBatch );

            if self.Holdout > 0
                % get the validation data (one-time only)
                [ dlXVal, dlYVal ] = thisValData.getDLInput( thisModel.XDimLabels );
            end

            % setup whole training set
            [ dlXTrnAll, dlYTrnAll ] = thisTrnData.getDLInput( thisModel.XDimLabels );

            % initialize counters
            nIter = iterationsPerEpoch( mbqTrn );           
            j = 0;
            v = 0;
            vp = self.ValPatience;
            
            % initialize logs
            nTrnLogs = nIter*self.NumEpochs;
            nValLogs = max( ceil( (self.NumEpochs-self.NumEpochsPreTrn) ...
                                        /self.ValFreq ), 1 );
            self.LossTrn = zeros( nTrnLogs, thisModel.NumLoss );
            self.LossVal = zeros( nValLogs, 1 );

            nMetricLogs = max( nTrnLogs/(nIter*self.UpdateFreq), 1 );
            self.Metrics = table( ...
                zeros( nMetricLogs, 1 ), ...
                zeros( nMetricLogs, 1 ), ...
                zeros( nMetricLogs, 1 ), ...
                zeros( nMetricLogs, 1 ), ...
                zeros( nMetricLogs, thisModel.ZDim ), ...
                VariableNames = {'ZCorrelation', 'XCCorrelation', ...
                                 'ZCovariance', 'XCCovariance', ...
                                 'VarianceProportion'} );


            for epoch = 1:self.NumEpochs
                
                self.CurrentEpoch = epoch;

                % Pre-training
                self.PreTraining = (epoch<=self.NumEpochsPreTrn);
                if self.PreTraining
                    nLoss = 1;
                else
                    nLoss = thisModel.NumLoss;
                end
            
                if thisTrnData.isFixedLength && self.HasMiniBatchShuffle
                    
                    % reset with a shuffled order
                    if self.HasShuffleRandomStream
                        % switch random streams for shuffling
                        modelRandomState = rng;
                        if epoch > 1
                            rng( shuffleRandomState );
                        end
                    end

                    shuffle( mbqTrn );
                    
                    if self.HasShuffleRandomStream
                        % switch back to the model random stream
                        shuffleRandomState = rng;
                        rng( modelRandomState );  
                    end
                
                else
                
                    % reset whilst preserving the order
                    reset( mbqTrn );
                
                end
            
                % loop over mini-batches
                for i = 1:nIter
                    
                    j = j + 1;
                    
                    % read mini-batch of data
                    [ dlXTTrn, dlXNTrn, dlYTrn ] = next( mbqTrn );
                    
                    % evaluate the model gradients 
                    [ grads, states, self.LossTrn(j,1:nLoss) ] = ...
                                      dlfeval(  @gradients, ...
                                                thisModel.Nets, ...
                                                thisModel, ...
                                                dlXTTrn, ...
                                                dlXNTrn, ...
                                                dlYTrn, ...
                                                self.PreTraining );

                    % store revised network states
                    for m = 1:thisModel.NumNetworks
                        thisName = thisModel.NetNames{m};
                        if isfield( states, thisName )
                            thisModel.Nets.(thisName).State = states.(thisName);
                        end
                    end

                    % update network parameters
                    [ thisModel.Optimizer, thisModel.Nets ] = ...
                        thisModel.Optimizer.updateNets( thisModel.Nets, ...
                                                        grads, ...
                                                        j );

                    if self.ShowPlots
                        % update loss plots
                        updateLossLines( self.LossLines, j, self.LossTrn(j,:) );
                    end

                end
                               
                if ~self.PreTraining ...
                        && mod( epoch, self.ValFreq )==0 ...
                        && self.Holdout > 0
                    
                    % run a validation check
                    v = v + 1;
                   
                    % compute relevant loss
                    self.LossVal(v) = validationCheck( thisModel, ...
                                                    self.ValType, ...
                                                    dlXVal, dlYVal );
                    if v > 2*vp-1
                        if mean(self.LossVal(v-2*vp+1:v-vp)) ...
                                < mean(self.LossVal(v-vp+1:v))
                            % no longer improving - stop training
                            break
                        end
                    end

                end
            
                % update progress on screen
                if mod( epoch, self.UpdateFreq )==0 && self.ShowPlots
                    
                    if ~self.PreTraining && self.Holdout > 0 && v > 0
                        % include validation
                        lossValArg = self.LossVal( v );
                    else
                        % exclude validation
                        lossValArg = [];
                    end

                    % record relevant metrics
                    [ self.Metrics( epoch/self.UpdateFreq, : ), ...
                        dlZTrnAll ] = calcMetrics( thisModel, dlXTrnAll );

                    % report 
                    self.reportProgress( thisModel, ...
                                         dlZTrnAll, ...
                                         thisTrnData.Y, ...
                                         self.LossTrn( j-nIter+1:j, : ), ...
                                         epoch, ...
                                         lossVal = lossValArg );
                end
            
                % update the number of dimensions, if required
                if mod( epoch, self.ActiveZFreq )==0
                    thisModel = thisModel.incrementActiveZDim;
                end

                if mod( epoch, self.LRFreq )==0
                    % update learning rates
                    thisModel.Optimizer = ...
                        thisModel.Optimizer.updateLearningRates( self.PreTraining );
                end

            end


            % train the auxiliary model
            dlZTrnAll = thisModel.encode( dlXTrnAll, convert = false );
            thisModel.AuxModel = trainAuxModel( ...
                                        thisModel.AuxModelType, ...
                                        dlZTrnAll, ...
                                        dlYTrnAll );


        end

            
    end      


    methods (Static)

        function reportProgress( thisModel, dlZ, dlY, ...
                       lossTrn, epoch, args )
            % Report progress on training
            arguments
                thisModel       CompactAEModel
                dlZ             dlarray
                dlY             dlarray
                lossTrn         double
                epoch           double
                args.nLines     double = 8
                args.lossVal    double = []
            end
        
            if size( lossTrn, 1 ) > 1
                meanLoss = mean( lossTrn );
            else
                meanLoss = lossTrn;
            end
        
            fprintf('Loss (%4d) = ', epoch);
            for k = 1:thisModel.NumLoss
                fprintf(' %6.3f', meanLoss(k) );
            end
            if ~isempty( args.lossVal )
                fprintf(' : %1.3f', args.lossVal );
            else
                fprintf('        ');
            end
            fprintf('\n');

            % compute the AE components for plotting
            [ dlXC, dlXMean ] = thisModel.calcLatentComponents( ...
                                            dlZ, ...
                                            sampling = 'Fixed' );

            % plot them on specified axes
            plotLatentComp( thisModel, ...
                            XMean = dlXMean, XC = dlXC, ...
                            type = 'Smoothed', shading = true );
        
            % plot the Z distributions
            plotZDist( thisModel, dlZ );
        
            % plot the Z clusters
            plotZClusters( thisModel, dlZ, Y = dlY );
        
            drawnow;
              
        end


    end


end


function [grad, state, loss] = gradients( nets, ...
                                          thisModel, ...
                                          dlXIn, ...
                                          dlXOut, ... 
                                          dlY, ...
                                          preTraining )
    % Compute the model gradients
    % (Model object not supplied so nets can be traced)
    arguments
        nets         struct   % networks, separate for traceability
        thisModel    CompactAEModel % contains all other relevant info
        dlXIn        dlarray  % input to the encoder
        dlXOut       dlarray  % output target for the decoder
        dlY          dlarray  % auxiliary outcome variable
        preTraining  logical  % flag indicating if in pretraining mode
    end

    % autoencoder training
    [ dlZGen, state.Encoder, dlZMu, dlZLogVar ] = ...
            forwardEncoder( thisModel, nets.Encoder, dlXIn );

    [ dlXGen, state.Decoder ] = ...
            forwardDecoder( thisModel, nets.Decoder, dlZGen );

    if thisModel.IsVAE
        % duplicate X & Y to match VAE's multiple draws
        nDraws = size( dlXGen, 2 )/size( dlXOut, 2 );
        dlXOut = repmat( dlXOut, 1, nDraws );
        dlY = repmat( dlY, 1, nDraws );
    end

    % select the active loss functions
    if preTraining
        isActive = thisModel.LossFcnTbl.Types=='Reconstruction';
    else
        isActive = thisModel.LossFcnTbl.DoCalcLoss;
    end

    activeFcns = thisModel.LossFcnTbl( isActive, : );

    compLossFcnIdx = find( activeFcns.Types=='Component', 1 );
    if ~isempty( compLossFcnIdx )
        % identify the component loss function
        thisName = activeFcns.Names(compLossFcnIdx);
        thisLossFcn = thisModel.LossFcns.(thisName);
        % compute the AE components
        dlXC = thisModel.calcLatentComponents( ...
                                dlZGen, ...
                                nSample = thisLossFcn.NumSamples, ...
                                forward = true, ...
                                dlX = dlXIn );
    end

    
    % compute the active loss functions in turn
    % and assign to networks
    
    nFcns = size( activeFcns, 1 );
    nLoss = sum( activeFcns.NumLosses );
    loss = zeros( nLoss, 1 );
    idx = 1;
    lossAccum = [];
    for i = 1:nFcns
       
        % identify the loss function
        thisName = activeFcns.Names(i);
        thisLossFcn = thisModel.LossFcns.(thisName);

        % assign indices for the number of losses returned
        lossIdx = idx:idx+thisLossFcn.NumLoss-1;
        idx = idx + thisLossFcn.NumLoss;

        % select the input variables
        switch thisLossFcn.Input
            case 'X-XHat'
                dlV = { dlXOut, dlXGen };
            case 'XC'
                dlV = { dlXC };
            case 'XHat'
                dlV = { dlXGen };
            case 'Z'
                dlV = { dlZGen };
            case 'ZMu-ZLogVar'
                dlV = { dlZMu, dlZLogVar };
            case 'YHat'
                dlV = dlYHat;
            case 'Z-Y'
                dlV = { dlZGen, dlY };
            case 'X-Y'
                dlV = { dlXIn, dlY };
        end

        % calculate the loss
        % (make sure to use the model's copy 
        %  of the relevant network object)
        if thisLossFcn.HasNetwork
            % call the loss function with the network object
            thisNetwork = nets.(thisName);
            if thisLossFcn.HasState
                % and store the network state too
                [ thisLoss, state.(thisName) ] = ...
                        thisLossFcn.calcLoss( thisNetwork, dlV{:} );
            else
                thisLoss = thisLossFcn.calcLoss( thisNetwork, dlV{:} );
            end
        else
            % call the loss function straightforwardly
            thisLoss = thisLossFcn.calcLoss( dlV{:} );
        end
        loss( lossIdx ) = thisLoss;

        if thisLossFcn.UseLoss
            lossAccum = assignLosses( lossAccum, thisLossFcn, thisLoss, lossIdx );
        end

    end

    % compute the gradients for each network
    netNames = fieldnames( lossAccum );
    for i = 1:length(netNames)
    
        thisName = netNames{i};
        thisNetwork = nets.(thisName);
        grad.(thisName) = dlgradient( lossAccum.(thisName), ...
                                      thisNetwork.Learnables, ...
                                      'RetainData', true );
        
    end

end


function lossAccum = assignLosses( lossAccum, thisLossFcn, thisLoss, lossIdx )
    % Assign loss to loss accumulator for associated network(s)

    for j = 1:length( lossIdx )

        for k = 1:length( thisLossFcn.LossNets(j,:) )

            netAssignments = string(thisLossFcn.LossNets{j,k});

                for l = 1:length(netAssignments)

                    netName = netAssignments(l);
                    if exist( 'lossAccum', 'var' )
                        if isfield( lossAccum, netName )
                            lossAccum.(netName) = ...
                                        lossAccum.(netName) + thisLoss(j);
                        else
                            lossAccum.(netName) = thisLoss(j);
                        end
                    else
                        lossAccum.(netName) = thisLoss(j);
                    end
                    
                end

        end
    end


end



function [ metric, dlZ ] = calcMetrics( thisModel, dlX )
    % Compute various supporting metrics
    arguments
        thisModel   CompactAEModel
        dlX         dlarray
    end

    % generate full-set encoding
    dlZ = thisModel.encode( dlX, convert = false );

    % record latent codes correlation
    [ metric.ZCorrelation, metric.ZCovariance ] = ...
                    latentCodeCorrelation( dlZ, summary = true );

    % generate validation components
    [ dlXC, ~, offsets ] = thisModel.calcLatentComponents( ...
                    dlZ, ...
                    nSample = 100, ...
                    sampling = 'Random' );

    % record latent codes correlation
    [ metric.XCCorrelation, metric.XCCovariance ] = ...
                latentComponentCorrelation( dlXC, 100, summary = true );
    metric.XCCorrelation = mean( metric.XCCorrelation );
    metric.XCCovariance = mean( metric.XCCovariance );

    % compute explained variance
    metric.VarianceProportionXC = ...
                        thisModel.explainedVariance( dlX, dlXC, offsets );

    % convert to table
    metric = struct2table( metric );

end


function model = trainAuxModel( modelType, dlZTrn, dlYTrn )
    % Train a non-network auxiliary model
    arguments
        modelType   string ...
            {mustBeMember(modelType, {'Logistic', 'Fisher', 'SVM'} )}
        dlZTrn      dlarray
        dlYTrn      dlarray
    end
    
    % convert to double for models which don't take dlarrays
    ZTrn = double(extractdata( dlZTrn ))';
    YTrn = double(extractdata( dlYTrn ));
    
    % fit the appropriate model
    switch modelType
        case 'Logistic'
            model = fitclinear( ZTrn, YTrn, Learner = "logistic" );
        case 'Fisher'
            model = fitcdiscr( ZTrn, YTrn );
        case 'SVM'
            model = fitcecoc( ZTrn, YTrn );
    end

end
 

function lossVal = validationCheck( thisModel, valType, dlXVal, dlYVal )
    % Validate the model so far
    arguments
        thisModel       CompactAEModel
        valType         string
        dlXVal          dlarray
        dlYVal          dlarray
    end

    dlZVal = thisModel.encode( dlXVal, convert = false );

    switch valType
        case 'Reconstruction'
            dlXValHat = squeeze(thisModel.reconstruct( dlZVal, convert = false ));
            lossVal = reconLoss( dlXVal, dlXValHat, thisModel.Scale );

        case 'AuxNetwork'
            dlYHatVal = predict( thisModel.Nets.(thisModel.auxNetwork), dlZVal );
            cLabels = thisModel.LossFcns.(thisModel.auxNetwork).CLabels;
            dlYHatVal = double( onehotdecode( dlYHatVal, single(cLabels), 1 ));
            lossVal = sum( dlYHatVal~=dlYVal )/length(dlYVal);

        case 'AuxModel'
            ZVal = double(extractdata( dlZVal ));
            YVal = double(extractdata( dlYVal ));
            lossVal = loss( thisModel.AuxModel, ZVal', YVal );
    end


end


function i = iterationsPerEpoch( mbq )
    % Count the number of iterations per epoch

    reset( mbq );
    i = 0;
    while hasdata( mbq )
        next( mbq );
        i = i+1;
    end

end



function [fig, lossLines] = initializeLossPlots( lossFcnTbl )
    % Setup plots for tracking progress
   
    nAxes = size( lossFcnTbl, 1 );
    [ rows, cols ] = sqdim( nAxes );
    
    % setup figure for plotting loss functions
    fig = figure(3);
    clf;
    nLines = sum( lossFcnTbl.NumLosses );
    lossLines = gobjects( nLines, 1 );
    colours = lines( nLines );
    c = 0;
    for i = 1:nAxes
        thisName = lossFcnTbl.Names(i);
        axis = subplot( rows, cols, i );
        for k = 1:lossFcnTbl.NumLosses(i)
            c = c+1;
            lossLines(c) = animatedline( axis, 'Color', colours(c,:) );
        end
        title( axis, thisName );
        xlabel( axis, 'Iteration' );
    end

end


function updateLossLines( lossLines, j, newPts )
    % Update loss animated lines
    arguments
        lossLines
        j               double
        newPts          double
    end

    for i = 1:length(lossLines)
        addpoints( lossLines(i), j, newPts(i) );
    end
    drawnow;

end



