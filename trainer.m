classdef trainer 
    % Class defining a model trainer

    properties
        networks       % networks for training
        modelNames     % names of those networks
        optimizer      % name of the optimizer to use
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
    end

    methods

        function self = trainer( networks, padValue, padLoc, args )
            % Initialize the model
            arguments
                networks            struct
                padValue
                padLoc              char ...
                    {mustBeMember(padLoc, ...
                      {'left', 'right', 'both', 'symmetric'} )}
                args.optimizer      char ...
                    {mustBeMember(args.optimizer, {'ADAM', 'SGD'} )} = 'ADAM'
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
                args.learningRates  double ...
                    {mustBePositive} = 0.001;
                args.lrFreq         double ...
                    {mustBeInteger, mustBePositive} = 250;
                args.lrFactor       double ...
                    {mustBeInteger, mustBePositive} = 0.5;
                args.valPatience    double ...
                    {mustBeInteger, mustBePositive} = 25;
                args.postTraining   logical = true;
                args.valType        char ...
                    {mustBeMember(args.valType, {'Network', 'Fisher'} )} = 'Fisher'

            end

            % initialize the training parameters
            self.optimizer.name = args.optimizer;
            self.nEpochs = args.nEpochs;
            self.nEpochsPreTrn = args.nEpochsPreTrn;
            self.batchSize = args.batchSize;
            self.valFreq = args.valFreq;
            self.updateFreq = args.updateFreq;
            self.lrFreq = args.lrFreq;
            self.valPatience = args.valPatience;

            self.preTraining = true;
            self.postTraining = args.postTraining;

            self.padValue = padValue;
            self.padLoc = padLoc;

            % initialize the optimizer's variables
            self.networks = networks;
            self.modelNames = fieldnames( self.networks );

            nModels = length( self.modelNames );
            if length( args.learningRates ) > 1
                if length( arg.learningRates ) == nModels
                    netSpecific = true;
                else
                    eid = 'Trainer:LearningRateMisMatch';
                    msg = 'The number of learning rates does not match the number of networks.';
                    throwAsCaller( MException(eid,msg) );
                end
            else
                netSpecific = false;
            end
            
            for i = 1:nModels
                networkName = self.modelNames{i};
                if netSpecific
                    self.learningRates.(networkName) = args.learningRates(i);
                else
                    self.learningRates.(networkName) = args.learningRates;
                end
                switch self.optimizer.name
                    case 'ADAM'
                        self.optimizer.(networkName).avgG = []; 
                        self.optimizer.(networkName).avgGS = [];
                    case 'SGD'
                        self.optimizer.(networkName).vel = [];
                end
            end

        end


        function model = train( self, model, X, XN, Y )
            arguments
                self
                model    aeModel
                X        double
                XN       double
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
                                  'MiniBatchFormat', {'CB', XNfmt, 'CB'} );
            end

            % setup the loop
            nIter = floor( size(XNTrn,2)/self.batchSize );
            nModels = length( self.modelNames );
            
            j = 0;
            v = 0;
            vp = self.valPatience;
            
            lossTrn = zeros( nIter*self.nEpochs, 10 );
            lossVal = zeros( ceil(nIter*self.nEpochs/self.valFreq), 1 );
            
            for epoch = 1:self.nEpochs
                
                % Pre-training
                self.preTraining = (epoch<=self.nEpochsPreTrn);
            
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
                                      dlfeval(  @modelGradients, ...
                                                models, ...
                                                dlXTTrn, ...
                                                dlXNTrn, ...
                                                dlYTrn, ...
                                                trainer );

                    % update the network parameters
                    for m = 1:nModels

                        net = self.networks.(self.modelNames{i});
                        opt = self.optimizer.(self.modelNames{i});

                        try
                            net.State = state.enc;
                        catch
                            lossTrn = NaN;
                            constraint = 2;
                            return
                        end

                        if any(strcmp( net, {'encoder','decoder'} )) ...
                            && not(self.postTraining || self.preTraining)
                            % skip training for the AE
                            continue
                        end

                        % update the network parameters
                        switch self.optimizer
                            case 'ADAM'         
                                [ net, opt.avgG, opt.avgGS ] = ...
                                        adamupdate( net, ...
                                                    grad, ...
                                                    opt.avgG, ...
                                                    opt.avgGS, ...
                                                    j, ...
                                                    self.learnRate, ...
                                                    self.beta1, ...
                                                    self.beta2 );
                            case 'SGD'
                                [ net, opt.vel ] = sgdmupdate( net, ...
                                            grad, ...
                                            opt.vel, ...
                                            self.dec.learnRate );
                        end
                        
                        self.networks.(self.modelNames{i}) = net;
                        self.optimizer.(self.modelNames{i}) = opt;
                    
                    end

                end
               

                if ~self.preTraining && mod( epoch, self.valFreq )==0
                    
                    % run a validation check
                    v = v + 1;
                    lossVal( v ) = validationCheck( model, ...
                                                dlXVal, dlYVal, cLabels );
                    if v > 2*vp-1
                        if mean(lossVal(v-2*vp+1:v-vp)) < mean(lossVal(v-vp+1:v))
                            % no longer improving - stop training
                            break
                        end
                    end

                end
            
                end

                % update progress on screen
                if mod( epoch, self.updateFreq )==0
                    meanLoss = mean(lossTrn( j-nIter+1:j, : ));
                    fprintf('Loss (%4d) = %6.3f  %1.3f  %1.3f %1.3f  %1.3f  %1.3f  %1.3f  %1.3f  %1.3f  %1.3f', ...
                        epoch, meanLoss );
                    if self.preTraining
                        fprintf('\n');
                    else
                        fprintf(' : %1.3f\n', lossVal(v));
                    end
            
                    dlZTrn = encode( model, dlXTTrn );
                    ZTrn = double(extractdata( dlZTrn ));
                    for c = 1:self.XChannels
                        plotLatentComp( ax.ae.comp(:,c), dlnetDec, ZTrn, c, ...
                                        self.fda.tSpan, self.fda.fdPar );
                    end
                    plotZDist( ax.ae.distZTrn, ZTrn, 'AE: Z Train', true );
                    drawnow;
                end
            
                if mod( epoch, self.lrFreq )==0
                    % update learning rates
                    for m = 1:nModels
                        if any(strcmp( net, {'encoder','decoder'} )) ...
                            && not(self.postTraining || self.preTraining)
                            % skip training for the AE
                            continue
                        end
                        self.learnRates.(self.modelName{i}) = ...
                            self.learnRates.(self.modelName{i})*self.lrFactor;
                    end
                end

            end

           
        end


    methods (Static)

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


        function [ dsTrn, XNfmt ] = createDataStore( XTrn, XNTrn, YTrn )

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


        function lossVal = validationCheck( model, dlXVal, dlYVal, cLabels )

            dlZVal = encode( model, dlXVal );
            switch self.valType
                case 'Network'
                    dlYHatVal = predict( dlnetCls, dlZVal );
                    dlYHatVal = double( ...
                        onehotdecode( dlYHatVal, single(cLabels), 1 ))' - 1;
                    lossVal(v) = sum( dlYHatVal~=dlYVal )/length(dlYVal);
    
                case 'Fisher'
                    ZVal = double(extractdata( dlZVal ));
                    YVal = double(extractdata( dlYVal ));
                    model = fitcdiscr( ZVal', YVal );
                    lossVal(v) = loss( model, ZVal', YVal );
            end
    

        end



    end

end



