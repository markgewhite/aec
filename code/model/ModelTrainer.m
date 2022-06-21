classdef ModelTrainer < handle
    % Class defining a model trainer

    properties
        NumEpochs        % maximum number of epochs for training
        NumEpochsPreTrn  % number of epochs for pretraining
        CurrentEpoch     % epoch counter
        BatchSize        % minibatch size
        PartialBatch     % what to do with an incomplete batch

        Holdout          % proportion of the dataset for validation
        ValFreq          % validation frequency in epochs
        UpdateFreq       % update frequency in epochs
        LRFreq           % learning rate update frequency

        ValPatience      % validation patience in valFreq units
        ValType          % validation function name

        NumLossFcns      % number of loss functions
        LossTrn          % record of training losses
        LossVal          % record of validation losses

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
                args.numEpochs      double ...
                    {mustBeInteger, mustBePositive} = 2000;
                args.numEpochsPreTrn  double ...
                    {mustBeInteger, mustBePositive} = 10;
                args.batchSize      double ...
                    {mustBeInteger, mustBePositive} = 40;
                args.partialBatch   char ...
                    {mustBeMember(args.partialBatch, ...
                        {'discard', 'return'} )} = 'discard'
                args.holdout        double ...
                    {mustBeInRange( args.holdout, 0, 0.5 )} = 0.2
                args.valFreq        double ...
                    {mustBeInteger, mustBePositive} = 5; 
                args.updateFreq     double ...
                    {mustBeInteger, mustBePositive} = 50;
                args.lrFreq         double ...
                    {mustBeInteger, mustBePositive} = 150;
                args.valPatience    double ...
                    {mustBeInteger, mustBePositive} = 25;
                args.postTraining   logical = true;
                args.valType        char ...
                    {mustBeMember(args.valType, ...
                        {'Reconstruction', 'AuxNetwork', 'AuxModel'} )} ...
                            = 'Reconstruction'
                args.showPlots      logical = false

            end

            % initialize the training parameters
            self.NumEpochs = args.numEpochs;
            self.NumEpochsPreTrn = args.numEpochsPreTrn;
            self.CurrentEpoch = 0;
            self.BatchSize = args.batchSize;
            self.PartialBatch = args.partialBatch;

            self.Holdout = args.holdout;
            self.ValFreq = args.valFreq;
            self.UpdateFreq = args.updateFreq;
            self.LRFreq = args.lrFreq;

            self.ValPatience = args.valPatience;
            self.ValType = args.valType;

            self.NumLossFcns = size( lossFcnTbl,1 );
            self.LossTrn = [];
            self.LossVal = [];

            self.PreTraining = true;
            self.PostTraining = args.postTraining;

            self.ShowPlots = args.showPlots;

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
            
            self.LossTrn = zeros( nIter*self.NumEpochs, thisModel.NumLoss );
            self.LossVal = zeros( ceil( (self.NumEpochs-self.NumEpochsPreTrn) ...
                                        /self.ValFreq ), 1 );
            
            for epoch = 1:self.NumEpochs
                
                self.CurrentEpoch = epoch;

                % Pre-training
                self.PreTraining = (epoch<=self.NumEpochsPreTrn);
                doTrainAE = (self.PostTraining || self.PreTraining);
            
                if thisTrnData.isFixedLength
                    % reset with a shuffled order
                    shuffle( mbqTrn );
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
                    [ grads, states, self.LossTrn(j,:) ] = ...
                                      dlfeval(  @gradients, ...
                                                thisModel.Nets, ...
                                                thisModel, ...
                                                dlXTTrn, ...
                                                dlXNTrn, ...
                                                dlYTrn, ...
                                                doTrainAE );

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
                                                        j, ...
                                                        doTrainAE );

                    if self.ShowPlots
                        % update loss plots
                        updateLossLines( self.LossLines, j, self.LossTrn(j,:) );
                    end

                end

                % train the auxiliary model
                dlZTrnAll = thisModel.encode( dlXTrnAll, ...
                                              convert = false );

                thisModel.AuxModel = trainAuxModel( ...
                                            thisModel.AuxModelType, ...
                                            dlZTrnAll, ...
                                            dlYTrnAll );
               

                if ~self.PreTraining ...
                        && mod( epoch, self.ValFreq )==0 ...
                        && self.Holdout > 0
                    
                    % run a validation check
                    v = v + 1;
                    self.LossVal( v ) = validationCheck( thisModel, ...
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
                    if ~self.PreTraining && self.Holdout > 0
                        % include validation
                        lossValArg = self.LossVal( v );
                    else
                        % exclude validation
                        lossValArg = [];
                    end
                     
                    self.reportProgress( thisModel, ...
                                         thisTrnData, ...
                                         self.LossTrn( j-nIter+1:j, : ), ...
                                         epoch, ...
                                         lossVal = lossValArg );
                end
            
                if mod( epoch, self.LRFreq )==0
                    % update learning rates
                    thisModel.Optimizer = ...
                        thisModel.Optimizer.updateLearningRates( doTrainAE );
                end

            end


        end


    end


    methods (Static)

        function reportProgress( thisModel, thisData, ...
                       lossTrn, epoch, args )
            % Report progress on training
            arguments
                thisModel       CompactAEModel
                thisData        ModelDataset
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
        
            [dlX, dlY] = thisData.getDLInput( thisModel.XDimLabels );
        
            % generate the latent encodings
            dlZ = thisModel.encode( dlX, convert = false );

            % compute the AE components for variance calculation
            [ dlXC, offsets ] = thisModel.latentComponents( ...
                            dlZ, ...
                            nSample = 10, ...
                            sampling = 'Fixed', ...
                            centre = false );

            % compute explained variance
            varProp = thisModel.explainedVariance( dlX, dlXC, offsets );
            fprintf('; VarProp = %5.3f (', sum(varProp) );
            for k = 1:length(varProp)
                fprintf(' %5.3f', varProp(k) );
            end
            fprintf(' )\n');


            % compute the AE components for plotting
            dlXC = thisModel.latentComponents( ...
                            dlZ, ...
                            sampling = 'Fixed', ...
                            centre = false );

            % plot them on specified axes
            plotLatentComp( thisModel, XC = dlXC, ...
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
                                          doTrainAE )
    % Compute the model gradients
    % (Model object not supplied so nets can be traced)
    arguments
        nets         struct   % networks, separate for traceability
        thisModel    CompactAEModel % contains all other relevant info
        dlXIn        dlarray  % input to the encoder
        dlXOut       dlarray  % output target for the decoder
        dlY          dlarray  % auxiliary outcome variable
        doTrainAE    logical  % whether to train the AE
    end

   
    if doTrainAE
        % autoencoder training
        [ dlXGen, dlZGen, state ] = ...
                forward( thisModel, nets.Encoder, nets.Decoder, dlXIn );
    
        if thisModel.IsVAE
            % duplicate X & Y to match VAE's multiple draws
            nDraws = size( dlXGen, 2 )/size( dlXOut, 2 );
            dlXOut = repmat( dlXOut, 1, nDraws );
            dlY = repmat( dlY, 1, nDraws );
        end
        
    else
        % no autoencoder training
        dlZGen = predict( nets.Encoder, dlXIn );
    
    end


    % select the active loss functions
    isActive = thisModel.LossFcnTbl.DoCalcLoss;
    activeFcns = thisModel.LossFcnTbl( isActive, : );

    compLossFcnIdx = find( activeFcns.Types=='Component', 1 );
    if ~isempty( compLossFcnIdx )
        % identify the component loss function
        thisName = activeFcns.Names(compLossFcnIdx);
        thisLossFcn = thisModel.LossFcns.(thisName);
        % compute the AE components
        dlXC = thisModel.latentComponents( ...
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
            dlXValHat = thisModel.reconstruct( dlZVal );
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


