classdef trainer 
    % Class defining a model trainer

    properties
        nNetworks      % number of networks for ease of reference
        optimizer      % optimizer to use, plus its variable structures
        nEpochs        % maximum number of epochs for training
        nEpochsPreTrn  % number of epochs for pretraining
        batchSize      % minibatch size

        valFreq        % validation frequency in epochs
        updateFreq     % update frequency in epochs
        valPatience    % validation patience in valFreq units
        valType        % validation function name

        learningRates  % learning rates for all networks
        lrFreq         % learning rate update frequency
        lrFactor       % learning rate reduction factor

        preTraining    % flag to indicate AE training
        postTraining   % flag to indicate whether to continue training

        padValue       % TEMPORARY: pad value
        padLoc         % TEMPORARY: pad location
        cLabels        % TEMPORARY: category labels
    end

    methods

        function self = trainer( thisModel, padValue, padLoc, cLabels, args )
            % Initialize the model
            arguments
                thisModel           autoencoderModel
                padValue            double
                padLoc              char ...
                    {mustBeMember(padLoc, ...
                      {'left', 'right', 'both', 'symmetric'} )}
                cLabels             categorical
                args.optimizer      char ...
                    {mustBeMember(args.optimizer, {'ADAM', 'SGD'} )} = 'ADAM'
                args.beta1          double ...
                    {mustBeNumeric, mustBePositive} = 0.9;
                args.beta2          double ...
                    {mustBeNumeric, mustBePositive} = 0.999;
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
                args.initLearningRates  double ...
                    {mustBePositive} = 0.001;
                args.lrFreq         double ...
                    {mustBeInteger, mustBePositive} = 150;
                args.lrFactor       double ...
                    {mustBeNumeric, mustBePositive} = 0.5;
                args.valPatience    double ...
                    {mustBeInteger, mustBePositive} = 25;
                args.postTraining   logical = true;
                args.valType        char ...
                    {mustBeMember(args.valType, ...
                        {'Reconstruction', 'Network', 'Fisher'} )} ...
                            = 'Reconstruction'

            end

            % initialize the training parameters
            self.optimizer.name = args.optimizer;
            self.nEpochs = args.nEpochs;
            self.nEpochsPreTrn = args.nEpochsPreTrn;
            self.batchSize = args.batchSize;
            self.valFreq = args.valFreq;
            self.updateFreq = args.updateFreq;
            self.lrFreq = args.lrFreq;
            self.lrFactor = args.lrFactor;
            self.valPatience = args.valPatience;

            self.preTraining = true;
            self.postTraining = args.postTraining;
            self.valType = args.valType;

            self.padValue = padValue;
            self.padLoc = padLoc;
            self.cLabels = cLabels;

            self.nNetworks = length( thisModel.netNames );

            % initialize the optimizer's variables
            if length( args.initLearningRates ) > 1
                if length( args.initLearningRates ) == self.nNetworks
                    netSpecific = true;
                else
                    eid = 'Trainer:LearningRateMisMatch';
                    msg = 'The number of learning rates does not match the number of networks.';
                    throwAsCaller( MException(eid,msg) );
                end
            else
                netSpecific = false;
            end
            
            for i = 1:self.nNetworks
                networkName = thisModel.netNames{i};
                if netSpecific
                    self.learningRates.(networkName) = args.initLearningRates(i);
                else
                    self.learningRates.(networkName) = args.initLearningRates;
                end
                switch self.optimizer.name
                    case 'ADAM'
                        self.optimizer.(networkName).avgG = []; 
                        self.optimizer.(networkName).avgGS = [];
                        self.optimizer.(networkName).beta1 = args.beta1;
                        self.optimizer.(networkName).beta2 = args.beta2;
                    case 'SGD'
                        self.optimizer.(networkName).vel = [];
                end
            end

        end


        function thisModel = train( self, thisModel, X, XN, Y )
            arguments
                self
                thisModel    autoencoderModel
                X            
                XN           double
                Y            
            end

            % re-partition the data to create training and validation sets
            cvPart = cvpartition( Y, 'Holdout', 0.25 );
            
            [ XTrn, XNTrn, YTrn ] = createTrnData( X, XN, Y, cvPart );
            [ dlXVal, dlYVal ] = createValData( X, Y, cvPart, ...
                                          self.padValue, self.padLoc );

            % create the mini-batch queue for batch processing
            [ dsTrn, XNfmt ] = createDatastore( XTrn, XNTrn, YTrn );

            % setup the minibatch queues
            if iscell( X )
                preproc = @( X, XN, Y ) preprocMiniBatch( X, XN, Y, ...
                                          self.padValue, self.padLoc );
                mbqTrn = minibatchqueue(  dsTrn,...
                                  'MiniBatchSize', self.batchSize, ...
                                  'PartialMiniBatch', 'return', ...
                                  'MiniBatchFcn', preproc, ...
                                  'MiniBatchFormat', {'CTB', XNfmt, 'CB'} );
            else
                mbqTrn = minibatchqueue( dsTrn,...
                                  'MiniBatchSize', self.batchSize, ...
                                  'PartialMiniBatch', 'discard', ...
                                  'MiniBatchFormat', {'CB', XNfmt, 'BC'} );
            end

            % setup the loop
            nIter = floor( size(XNTrn,2)/self.batchSize );           
            j = 0;
            v = 0;
            vp = self.valPatience;
            
            lossTrn = zeros( nIter*self.nEpochs, thisModel.nLoss );
            lossVal = zeros( ceil(nIter*self.nEpochs/self.valFreq), 1 );
            
            for epoch = 1:self.nEpochs
                
                % Pre-training
                self.preTraining = (epoch<=self.nEpochsPreTrn);
                doTrainAE = (self.postTraining || self.preTraining);
            
                if iscell( X )
                    % reset whilst preserving the order
                    reset( mbqTrn );
                else
                    % reset with a shuffled order
                    shuffle( mbqTrn );
                end
            
                % loop over mini-batches
                for i = 1:nIter
                    
                    j = j + 1;
                    
                    % read mini-batch of data
                    [ dlXTTrn, dlXNTrn, dlYTrn ] = next( mbqTrn );
                    if size( XNTrn, 3 ) > 1
                        dlXNTrn = dlarray( squeeze(dlXNTrn), 'SCB' );
                    end
                    
                    % evaluate the model gradients 
                    [ grad, state, lossTrn(j,:) ] = ...
                                      dlfeval(  @gradients, ...
                                                thisModel.nets, ...
                                                thisModel.lossFcns, ...
                                                thisModel.lossFcnTbl, ...
                                                dlXTTrn, ...
                                                dlXNTrn, ...
                                                dlYTrn, ...
                                                doTrainAE, ...
                                                thisModel.isVAE );

                    % update the network parameters
                    for m = 1:self.nNetworks

                        thisName = thisModel.netNames{m};
                        thisNetwork = thisModel.nets.(thisName);
                        thisOptimizer = self.optimizer.(thisName);
                        thisGradient = grad.(thisName);
                        thisLearningRate = self.learningRates.(thisName);


                        try
                            thisNetwork.State = state.(thisName);
                        catch
                            constraint = 2;
                            return
                        end

                        if any(strcmp( thisName, {'encoder','decoder'} )) ...
                            && not(self.postTraining || self.preTraining)
                            % skip training for the AE
                            continue
                        end

                        % update the network parameters
                        switch self.optimizer.name
                            case 'ADAM'         
                                [ thisNetwork, ...
                                  thisOptimizer.avgG, ...
                                  thisOptimizer.avgGS ] = ...
                                        adamupdate( thisNetwork, ...
                                                    thisGradient, ...
                                                    thisOptimizer.avgG, ...
                                                    thisOptimizer.avgGS, ...
                                                    j, ...
                                                    thisLearningRate, ...
                                                    thisOptimizer.beta1, ...
                                                    thisOptimizer.beta2 );
                            case 'SGD'
                                [ thisNetwork, ...
                                  thisOptimizer.vel ] = ...
                                    sgdmupdate( thisNetwork, ...
                                                thisGradient, ...
                                                thisOptimizer.vel, ...
                                                thisLearningRate );
                        end
                        
                        thisModel.nets.(thisName) = thisNetwork;
                        self.optimizer.(thisName) = thisOptimizer;
                    
                    end

                end
               

                if ~self.preTraining && mod( epoch, self.valFreq )==0
                    
                    % run a validation check
                    v = v + 1;
                    lossVal( v ) = validationCheck( thisModel, self.valType, ...
                                                        dlXVal, dlYVal );
                    if v > 2*vp-1
                        if mean(lossVal(v-2*vp+1:v-vp)) < mean(lossVal(v-vp+1:v))
                            % no longer improving - stop training
                            break
                        end
                    end

                end
            
                % update progress on screen
                if mod( epoch, self.updateFreq )==0
                    meanLoss = mean(lossTrn( j-nIter+1:j, : ));

                    fprintf('Loss (%4d) = ', epoch);
                    for k = 1:thisModel.nLoss
                        fprintf(' %6.3f', meanLoss(k) );
                    end
                    if self.preTraining
                        fprintf('\n');
                    else
                        fprintf(' : %1.3f\n', lossVal(v));
                    end
            
                    dlZTrn = thisModel.encode( thisModel, dlXTTrn );
                    %ZTrn = double(extractdata( dlZTrn ));
                    %for c = 1:self.XChannels
                    %    plotLatentComp( ax.ae.comp(:,c), dlnetDec, ZTrn, c, ...
                    %                    self.fda.tSpan, self.fda.fdPar );
                    %end
                    %plotZDist( ax.ae.distZTrn, ZTrn, 'AE: Z Train', true );
                    %drawnow;
                end
            
                if mod( epoch, self.lrFreq )==0
                    % update learning rates
                    for m = 1:self.nNetworks
                        thisName = thisModel.netNames{m};
                        if any(strcmp( thisName, {'encoder','decoder'} )) ...
                            && not(self.postTraining || self.preTraining)
                            % skip training for the AE
                            continue
                        end
                        self.learningRates.(thisName) = ...
                            self.learningRates.(thisName)*self.lrFactor;
                    end
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
            [ dlZGen, state.encoder ] = forward( nets.encoder, dlXIn);

        end

        % reconstruct curves from latent codes
        [ dlXGen, state.decoder ] = forward( nets.decoder, dlZGen );
        
    else
        % no autoencoder training
        dlZGen = predict( nets.encoder, dlXIn );
    
    end


    % select the active loss functions
    activeFcns = lossFcnInfo( lossFcnInfo.doCalcLoss, : );

    if any( activeFcns.types=='Component' )
        % compute the AE components
        if isVAE
            dlXC = latentComponents( nets.decoder, dlZGen, ...
                                        dlZMean = self.dlZMeans, ...
                                        dlZLogVar = self.dlZLogVars );
        else
            dlXC = latentComponents( nets.decoder, dlZGen );
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



function [ XTrn, XNTrn, YTrn ] = createTrnData( X, XN, Y, cvPart )

    if iscell( X )
        % X is a cell array containing sequences of variable length
        XTrn = X( training(cvPart) );

    else
        % X is a numeric array
        XTrn = X( :, training(cvPart) );

    end

    % create time normalisation set
    XNTrn = XN( :, training(cvPart), : );

    % create the outcome variable set
    YTrn = Y( training(cvPart) );

end


function [ dlXVal, dlYVal ] = createValData( X, Y, cvPart, padValue, padLoc )

    if iscell( X )
        % X is a cell array containing sequences of variable length
        XVal = preprocMiniBatch( X( test(cvPart) ), [], [], ...
                                 padValue, ...
                                 padLoc );
        dlXVal = dlarray( XVal, 'CTB' );

    else
        % X is a numeric array
        XVal = X( :, test(cvPart) );
        dlXVal = dlarray( XVal, 'CB' );

    end

    dlYVal = dlarray( Y( test(cvPart)  ), 'CB' );

end


function [ dsTrn, XNfmt ] = createDatastore( XTrn, XNTrn, YTrn )

    % create the datastore for the input X
    if iscell( XTrn )           
        % sort them in ascending order of length
        XLen = cellfun( @length, XTrn );
        [ ~, orderIdx ] = sort( XLen, 'descend' );
    
        XTrn = XTrn( orderIdx );
        dsXTrn = arrayDatastore( XTrn, 'IterationDimension', 1, ...
                                 'OutputType', 'same' );
    
    else
        dsXTrn = arrayDatastore( XTrn, 'IterationDimension', 2 );
    
    end
    
    % create the datastore for the time-normalised output X
    dsXNTrn = arrayDatastore( XNTrn, 'IterationDimension', 2 );
    if size( XNTrn, 3 ) > 1
        XNfmt = 'SSCB';
    else
        XNfmt = 'CB';
    end
    
    % create the datastore for the labels/outcomes
    dsYTrn = arrayDatastore( YTrn, 'IterationDimension', 1 );   
    
    % combine them
    dsTrn = combine( dsXTrn, dsXNTrn, dsYTrn );
               
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


