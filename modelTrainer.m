classdef modelTrainer < handle
    % Class defining a model trainer

    properties
        nEpochs        % maximum number of epochs for training
        nEpochsPreTrn  % number of epochs for pretraining
        currentEpoch   % epoch counter
        batchSize      % minibatch size
        partialBatch   % what to do with an incomplete batch

        valFreq        % validation frequency in epochs
        updateFreq     % update frequency in epochs
        lrFreq         % learning rate update frequency

        valPatience    % validation patience in valFreq units
        valType        % validation function name

        nLossFcns      % number of loss functions
        lossTrn        % record of training losses
        lossVal        % record of validation losses

        preTraining    % flag to indicate AE training
        postTraining   % flag to indicate whether to continue training

        showPlots      % flag whether to show plots
        lossFig        % figure for the loss lines
        lossLines      % animated lines cell array
    end

    methods

        function self = modelTrainer( lossFcnTbl, args )
            % Initialize the model
            arguments
                lossFcnTbl          table
                args.nEpochs        double ...
                    {mustBeInteger, mustBePositive} = 2000;
                args.nEpochsPreTrn  double ...
                    {mustBeInteger, mustBePositive} = 10;
                args.batchSize      double ...
                    {mustBeInteger, mustBePositive} = 40;
                args.partialBatch   char ...
                    {mustBeMember(args.partialBatch, ...
                        {'discard', 'return'} )} = 'discard'
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
            self.nEpochs = args.nEpochs;
            self.nEpochsPreTrn = args.nEpochsPreTrn;
            self.currentEpoch = 0;
            self.batchSize = args.batchSize;
            self.partialBatch = args.partialBatch;

            self.valFreq = args.valFreq;
            self.updateFreq = args.updateFreq;
            self.lrFreq = args.lrFreq;

            self.valPatience = args.valPatience;
            self.valType = args.valType;

            self.nLossFcns = size( lossFcnTbl,1 );
            self.lossTrn = [];
            self.lossVal = [];

            self.preTraining = true;
            self.postTraining = args.postTraining;

            self.showPlots = args.showPlots;

            if self.showPlots
                [self.lossFig, self.lossLines] = ...
                                initializeLossPlots( lossFcnTbl );
            end

        end

        
        function [ thisModel, thisOptimizer ] = runTraining( ...
                                                self, ...
                                                thisModel, ...
                                                thisTrnData, ...
                                                thisValData )
            % Run the training loop for the model
            arguments
                self            modelTrainer
                thisModel       autoencoderModel
                thisTrnData     modelDataset
                thisValData     modelDataset
            end

            
            % setup the minibatch queues
            mbqTrn = thisTrnData.getMiniBatchQueue( thisTrnData, ...
                                        self.batchSize, ...
                                        thisModel.XDimLabels, ...
                                        thisModel.XNDimLabels, ...
                                        partialBatch = self.partialBatch );

            % get the validation data (one-time only)
            [ dlXVal, dlYVal ] = thisValData.getDLInput( thisModel.XDimLabels );
            
            % setup whole training set
            [ dlXTrnAll, dlYTrnAll ] = thisTrnData.getDLInput( thisModel.XDimLabels );

            % initialize counters
            nIter = iterationsPerEpoch( mbqTrn );           
            j = 0;
            v = 0;
            vp = self.valPatience;
            
            self.lossTrn = zeros( nIter*self.nEpochs, thisModel.nLoss );
            self.lossVal = zeros( ceil( (self.nEpochs-self.nEpochsPreTrn) ...
                                        /self.valFreq ), 1 );
            
            for epoch = 1:self.nEpochs
                
                self.currentEpoch = epoch;

                % Pre-training
                self.preTraining = (epoch<=self.nEpochsPreTrn);
                doTrainAE = (self.postTraining || self.preTraining);
            
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
                    [ grads, states, self.lossTrn(j,:) ] = ...
                                      dlfeval(  @gradients, ...
                                                thisModel.nets, ...
                                                thisModel, ...
                                                dlXTTrn, ...
                                                dlXNTrn, ...
                                                dlYTrn, ...
                                                doTrainAE );

                    % store revised network states
                    for m = 1:thisModel.nNets
                        thisName = thisModel.netNames{m};
                        if isfield( states, thisName )
                            thisModel.nets.(thisName).State = states.(thisName);
                        end
                    end

                    % update network parameters
                    [ thisOptimizer, thisModel.nets ] = ...
                        thisModel.optimizer.updateNets( thisModel.nets, ...
                                                        grads, ...
                                                        j, ...
                                                        doTrainAE );

                    if self.showPlots
                        % update loss plots
                        updateLossLines( self.lossLines, j, self.lossTrn(j,:) );
                    end

                end

                % train the auxiliary model
                dlZTrnAll = thisModel.encode( thisModel, dlXTrnAll, ...
                                              convert = false );

                thisModel.auxModel = trainAuxModel( ...
                                            thisModel.auxModelType, ...
                                            dlZTrnAll, ...
                                            dlYTrnAll );
               

                if ~self.preTraining ...
                        && mod( epoch, self.valFreq )==0
                    
                    % run a validation check
                    v = v + 1;
                    self.lossVal( v ) = validationCheck( thisModel, ...
                                                    self.valType, ...
                                                    dlXVal, dlYVal );
                    if v > 2*vp-1
                        if mean(self.lossVal(v-2*vp+1:v-vp)) ...
                                < mean(self.lossVal(v-vp+1:v))
                            % no longer improving - stop training
                            break
                        end
                    end

                end
            
                % update progress on screen
                if mod( epoch, self.updateFreq )==0 && self.showPlots
                    if self.preTraining
                        % exclude validation
                        lossValArg = [];
                    else
                        % include validation
                        lossValArg = self.lossVal( v );
                    end                       
                    self.reportProgress( thisModel, ...
                                         thisTrnData, ...
                                         self.lossTrn( j-nIter+1:j, : ), ...
                                         epoch, ...
                                         lossVal = lossValArg );
                end
            
                if mod( epoch, self.lrFreq )==0
                    % update learning rates
                    thisOptimizer = ...
                        thisOptimizer.updateLearningRates( doTrainAE );
                end

            end


        end


    end


    methods (Static)

        function reportProgress( thisModel, thisData, ...
                       lossTrn, epoch, args )
            % Report progress on training
            arguments
                thisModel       autoencoderModel
                thisData        modelDataset
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
            for k = 1:thisModel.nLoss
                fprintf(' %6.3f', meanLoss(k) );
            end
            if ~isempty( args.lossVal )
                fprintf(' : %1.3f', args.lossVal );
            end
        
            [dlX, dlY] = thisData.getDLInput( thisModel.XDimLabels );
        
            % compute the AE components
            dlZ = thisModel.encode( thisModel, dlX, convert = false );
            dlXC = thisModel.latentComponents( ...
                            dlZ, ...
                            sampling = 'Fixed', ...
                            centre = false );

            % compute explained variance
            varProp = thisModel.getExplainedVariance( dlZ, thisData ); 
            fprintf('; VarProp = ');
            for k = 1:length(varProp)
                fprintf(' %8.2f', varProp(k) );
            end
            fprintf('\n');


            % plot them on specified axes
            thisModel.plotLatentComp( ...
                          dlXC, ...
                          thisData.tSpan, ...
                          thisData.fda.fdParamsTarget, ...
                          type = 'Smoothed', ...
                          shading = true, ...
                          plotTitle = thisData.info.datasetName, ...
                          xAxisLabel = thisData.info.timeLabel, ...
                          yAxisLabel = thisData.info.channelLabels, ...
                          yAxisLimits = thisData.info.channelLimits );
        
            % plot the Z distributions
            thisModel.plotZDist( dlZ );
        
            % plot the Z clusters
            thisModel.plotZClusters( dlZ, Y = dlY );
        
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
        thisModel    autoencoderModel % contains all other relevant info
        dlXIn        dlarray  % input to the encoder
        dlXOut       dlarray  % output target for the decoder
        dlY          dlarray  % auxiliary outcome variable
        doTrainAE    logical  % whether to train the AE
    end

   
    if doTrainAE
        % autoencoder training
        [ dlXGen, dlZGen, state ] = ...
                forward( thisModel, nets.encoder, nets.decoder, dlXIn );
    
        if thisModel.isVAE
            % duplicate X & Y to match VAE's multiple draws
            nDraws = size( dlXGen, 2 )/size( dlXOut, 2 );
            dlXOut = repmat( dlXOut, 1, nDraws );
            dlY = repmat( dlY, 1, nDraws );
        end
        
    else
        % no autoencoder training
        dlZGen = predict( nets.encoder, dlXIn );
    
    end


    % select the active loss functions
    isActive = thisModel.lossFcnTbl.doCalcLoss;
    activeFcns = thisModel.lossFcnTbl( isActive, : );

    compLossFcnIdx = find( activeFcns.types=='Component', 1 );
    if ~isempty( compLossFcnIdx )
        % identify the component loss function
        thisName = activeFcns.names(compLossFcnIdx);
        thisLossFcn = thisModel.lossFcns.(thisName);
        % compute the AE components
        dlXC = thisModel.latentComponents( ...
                                dlZGen, ...
                                nSample = thisLossFcn.NumSamples, ...
                                forward = true );
    end

    
    % compute the active loss functions in turn
    % and assign to networks
    
    nFcns = size( activeFcns, 1 );
    nLoss = sum( activeFcns.nLosses );
    loss = zeros( nLoss, 1 );
    idx = 1;
    lossAccum = [];
    for i = 1:nFcns
       
        % identify the loss function
        thisName = activeFcns.names(i);
        thisLossFcn = thisModel.lossFcns.(thisName);

        % assign indices for the number of losses returned
        lossIdx = idx:idx+thisLossFcn.nLoss-1;
        idx = idx + thisLossFcn.nLoss;

        % select the input variables
        switch thisLossFcn.input
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
        end

        % calculate the loss
        % (make sure to use the model's copy 
        %  of the relevant network object)
        if thisLossFcn.hasNetwork
            % call the loss function with the network object
            thisNetwork = nets.(thisName);
            if thisLossFcn.hasState
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

        if thisLossFcn.useLoss
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

        for k = 1:length( thisLossFcn.lossNets(j,:) )

            netAssignments = string(thisLossFcn.lossNets{j,k});

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
            {mustBeMember(modelType, {'Fisher', 'SVM'} )}
        dlZTrn      dlarray
        dlYTrn      dlarray
    end
    
    % convert to double for models which don't take dlarrays
    ZTrn = double(extractdata( dlZTrn ))';
    YTrn = double(extractdata( dlYTrn ));
    
    % fit the appropriate model
    switch modelType
        case 'Fisher'
            model = fitcdiscr( ZTrn, YTrn );
        case 'SVM'
            model = fitcecoc( ZTrn, YTrn );
    end

end
 

function lossVal = validationCheck( thisModel, valType, dlXVal, dlYVal )
    % Validate the model so far
    arguments
        thisModel       autoencoderModel
        valType         string
        dlXVal          dlarray
        dlYVal          dlarray
    end

    dlZVal = thisModel.encode( thisModel, dlXVal, convert = false );
    switch valType
        case 'Reconstruction'
            dlXValHat = thisModel.reconstruct( thisModel, dlZVal );
            lossVal = thisModel.getReconLoss( thisModel, dlXVal, dlXValHat );

        case 'AuxNetwork'
            dlYHatVal = predict( thisModel.nets.(thisModel.auxNetwork), dlZVal );
            cLabels = thisModel.lossFcns.(thisModel.auxNetwork).CLabels;
            dlYHatVal = double( onehotdecode( dlYHatVal, single(cLabels), 1 ));
            lossVal = sum( dlYHatVal~=dlYVal )/length(dlYVal);

        case 'AuxModel'
            ZVal = double(extractdata( dlZVal ));
            YVal = double(extractdata( dlYVal ));
            lossVal = loss( thisModel.auxModel, ZVal', YVal );
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
    nLines = sum( lossFcnTbl.nLosses );
    lossLines = gobjects( nLines, 1 );
    colours = lines( nLines );
    c = 0;
    for i = 1:nAxes
        thisName = lossFcnTbl.names(i);
        axis = subplot( rows, cols, i );
        for k = 1:lossFcnTbl.nLosses(i)
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


