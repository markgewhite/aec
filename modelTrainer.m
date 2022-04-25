classdef modelTrainer 
    % Class defining a model trainer

    properties
        nEpochs        % maximum number of epochs for training
        nEpochsPreTrn  % number of epochs for pretraining
        batchSize      % minibatch size

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
        axes           % plot axes structure
    end

    methods

        function self = modelTrainer( lossFcnTbl, XChannels, ZDim, args )
            % Initialize the model
            arguments
                lossFcnTbl          table
                XChannels           double
                ZDim                double
                args.nEpochs        double ...
                    {mustBeInteger, mustBePositive} = 2000;
                args.nEpochsPreTrn  double ...
                    {mustBeInteger, mustBePositive} = 10;
                args.batchSize      double ...
                    {mustBeInteger, mustBePositive} = 40;
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
                        {'Reconstruction', 'Network', 'Fisher'} )} ...
                            = 'Reconstruction'
                args.showPlots      logical = false

            end

            % initialize the training parameters
            self.nEpochs = args.nEpochs;
            self.nEpochsPreTrn = args.nEpochsPreTrn;
            self.batchSize = args.batchSize;

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
                self.axes = initializePlots( lossFcnTbl, XChannels, ZDim );
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
                                                self.batchSize );

            mbqVal = thisValData.getMiniBatchQueue( thisValData, ...
                                                thisValData.nObs );

            % initialize counters
            nIter = iterationsPerEpoch( mbqTrn );           
            j = 0;
            v = 0;
            vp = self.valPatience;

            % get the validation data (one-time only)
            [dlXVal, ~, dlYVal] = next( mbqVal ); 
            
            self.lossTrn = zeros( nIter*self.nEpochs, thisModel.nLoss );
            self.lossVal = zeros( ceil(nIter*self.nEpochs/self.valFreq), 1 );
            
            for epoch = 1:self.nEpochs
                
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
                    if size( dlXNTrn, 3 ) > 1
                        dlXNTrn = dlarray( squeeze(dlXNTrn), 'SCB' );
                    end
                    
                    % evaluate the model gradients 
                    [ grads, states, self.lossTrn(j,:) ] = ...
                                      dlfeval(  @gradients, ...
                                                thisModel.nets, ...
                                                thisModel.lossFcns, ...
                                                thisModel.lossFcnTbl, ...
                                                dlXTTrn, ...
                                                dlXNTrn, ...
                                                dlYTrn, ...
                                                @thisModel.latentComponents, ...
                                                doTrainAE, ...
                                                thisModel.isVAE );

                    % store revised network state
                    for m = 1:thisModel.nNets
                        thisName = thisModel.netNames{m};
                        try
                            thisModel.nets.(thisName).State = states.(thisName);
                        catch
                            constraint = 2;
                            return
                        end
                    end

                    % update network parameters
                    [ thisOptimizer, thisModel.nets ] = ...
                        thisModel.optimizer.updateNets( thisModel.nets, ...
                                                        grads, ...
                                                        j, ...
                                                        doTrainAE );

                    % update loss plots


                end
               

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
                if mod( epoch, self.updateFreq )==0
                    if self.preTraining
                        % exclude validation
                        reportProgress( self.axes, ...
                                    thisModel, thisTrnData, ...
                                    dlXTTrn, ...
                                    self.lossTrn( j-nIter+1:j, : ), ...
                                    epoch );
                    else
                        % include validation
                        reportProgress( self.axes, ...
                                    thisModel, thisTrnData, ...
                                    dlXTTrn, ...
                                    self.lossTrn( j-nIter+1:j, : ), ...
                                    epoch, ...
                                    lossVal = self.lossVal( v ) );
                    end
                end
            
                if mod( epoch, self.lrFreq )==0
                    % update learning rates
                    thisOptimizer = ...
                        thisOptimizer.updateLearningRates( doTrainAE );
                end

            end


        end

    end


end


function [grad, state, loss] = gradients( nets, ...
                                          lossFcns, ...
                                          lossFcnInfo, ...
                                          dlXIn, dlXOut, ... 
                                          dlY, ...
                                          compFcn, ...
                                          doTrainAE, ...
                                          isVAE )
    % Compute the model gradients
    % (Model object not supplied so nets can be traced)
    arguments
        nets         struct   % networks, made explicit for tracing
        lossFcns     struct   % loss functions
        lossFcnInfo  table    % supporting info on loss functions
        dlXIn        dlarray  % input to the encoder
        dlXOut       dlarray  % output target for the decoder
        dlY          dlarray  % auxiliary outcome variable
        compFcn               % latent components generating function
        doTrainAE    logical  % whether to train the AE
        isVAE        logical  % if variational autoencoder
    end

   
    if doTrainAE
        % autoencoder training
    
        if isVAE
            % generate latent encodings
            [ dlZGen, state.encoder, dlZMu, dlLogVar ] = ...
                forward( nets.encoder, dlXIn);

            % duplicate X & C to reflect mulitple draws of VAE
            dlXOut = repmat( dlXOut, 1, nets.encoder.nDraws );
            dlY = repmat( dlY, 1, nets.encoder.nDraws );
        
        else
            % generate latent encodings
            [ dlZGen, state.encoder ] = forward( nets.encoder, dlXIn );

        end

        % reconstruct curves from latent codes
        [ dlXGen, state.decoder ] = forward( nets.decoder, dlZGen );
        
    else
        % no autoencoder training
        dlZGen = predict( nets.encoder, dlXIn );
    
    end


    % select the active loss functions
    activeFcns = lossFcnInfo( lossFcnInfo.doCalcLoss, : );

    compLossFcnIdx = find( activeFcns.types=='Component', 1 );
    if ~isempty( compLossFcnIdx )
        % identify the component loss function
        thisName = activeFcns.names(compLossFcnIdx);
        thisLossFcn = lossFcns.(thisName);
        % compute the AE components
        if isVAE
            dlXC = compFcn( nets.decoder, dlZGen, ...
                            nSample = thisLossFcn.nSample, ...
                            dlZMean = dlZMu, ...
                            dlZLogVar = dlZLogVar );
        else
            dlXC = compFcn( nets.decoder, dlZGen, ...
                            nSample = thisLossFcn.nSample );
        end
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
        thisLossFcn = lossFcns.(thisName);

        % assign indices for the number of losses returned
        lossIdx = idx:idx+thisLossFcn.nLoss-1;
        idx = idx + thisLossFcn.nLoss;

        % select the input variables
        switch thisLossFcn.input
            case 'X-XHat'
                dlV = { dlXOut, dlXGen };
            case 'XC'
                dlV = { dlXC };
            case 'Z'
                dlV = { dlZGen };
            case 'Z-ZHat'
                dlV = { dlZGen, dlZReal };
            case 'ZMu-ZLogVar'
                dlV = { dlZMu, dlLogVar };
            case 'Y'
                dlV = dlY;
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
                [ thisLossFcn, thisLoss, state.(thisName) ] = ...
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
    netNames = fieldnames( nets );
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


function lossVal = validationCheck( thisModel, valType, dlXVal, dlYVal )

    dlZVal = thisModel.encode( thisModel, dlXVal );
    switch valType
        case 'Reconstruction'
            dlXValHat = thisModel.reconstruct( thisModel, dlZVal );
            lossVal = thisModel.getReconLoss( thisModel, dlXVal, dlXValHat );

        case 'Network'
            dlYHatVal = predict( thisModel.nets.classifier, dlZVal );
            dlYHatVal = double( ...
                onehotdecode( dlYHatVal, single(thisModel.cLabels), 1 ))' - 1;
            lossVal = sum( dlYHatVal~=dlYVal )/length(dlYVal);

        case 'Fisher'
            ZVal = double(extractdata( dlZVal ));
            YVal = double(extractdata( dlYVal ));
            classifier = fitcdiscr( ZVal', YVal );
            lossVal = loss( classifier, ZVal', YVal );
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



function axes = initializePlots( lossFcnTbl, XChannels, ZDim )
    % Setup plots for tracking progress
   
    nAxes = size( lossFcnTbl, 1 );
    [ rows, cols ] = sqdim( nAxes );
    
    % setup figure for plotting loss functions
    figure(1);
    for i = 1:nAxes
        axes.loss.(lossFcnTbl.names(i)) = subplot( rows, cols, i );
    end

    % setup figure for Z distribution and clustering
    figure(2);
    axes.ZDistribution = subplot( 1, 2, 1 );
    axes.ZClustering = subplot( 1, 2, 2 );

    % setup the components figure
    figure(3);
    axes.comp = gobjects( ZDim, XChannels );
    
    for j = 1:XChannels
        for i = 1:ZDim
            axes.comp(i,j) = subplot( XChannels, ZDim, (j-1)*ZDim + i );
        end
    end

end


function reportProgress( axes, thisModel, thisData, dlXTTrn, lossTrn, epoch, args )
    % Report progress on training
    arguments
        axes
        thisModel       autoencoderModel
        thisData        modelDataset
        dlXTTrn         dlarray
        lossTrn         double
        epoch           double
        args.nLines     double = 8
        args.lossVal    double = []
    end

    meanLoss = mean( lossTrn );

    fprintf('Loss (%4d) = ', epoch);
    for k = 1:thisModel.nLoss
        fprintf(' %6.3f', meanLoss(k) );
    end
    if isempty( args.lossVal )
        fprintf('\n');
    else
        fprintf(' : %1.3f\n', args.lossVal );
    end

    % compute the AE components
    dlZTrn = thisModel.encode( thisModel, dlXTTrn );
    dlXC = thisModel.latentComponents( ...
                    thisModel.nets.decoder, ...
                    dlZTrn, ...
                    sampling = 'Fixed', ...
                    nSample = args.nLines, ...
                    centre = false );

    % plot them on specified axes
    thisModel.plotLatentComp( axes.comp, ...
                  dlXC, ...
                  thisData.fda, ...
                  args.nLines, ...
                  type = 'Smoothed', ...
                  shading = true, ...
                  plotTitle = thisData.info.datasetName, ...
                  xAxisLabel = thisData.info.timeLabel, ...
                  yAxisLabel = thisData.info.channelLabels, ...
                  yAxisLimits = thisData.info.channelLimits );

    % plot the Z distributions
    thisModel.plotZDist( axes.ZDistribution, dlZTrn );

    drawnow;



end

