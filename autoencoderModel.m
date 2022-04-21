% ************************************************************************
% Class: autoencoderModel
%
% Subclass defining the framework for an autoencoder model
%
% ************************************************************************

classdef autoencoderModel < representationModel

    properties
        XOutDim        % output dimension (may differ from XDim)
        nets           % networks defined in this model (structure)
        netNames       % names of the networks (for convenience)
        nNets          % number of networks
        isVAE          % flag indicating if variational autoencoder
        lossFcns       % array of loss functions
        lossFcnNames   % names of the loss functions
        lossFcnWeights % weights to be applied to the loss function
        lossFcnTbl     % convenient table summarising loss function details
        nLoss          % number of computed losses
        isInitialized  % flag indicating if fully initialized
        hasSeqInput    % supports variable-length input

        trainer        % trainer object holding training parameters
        optimizer      % optimizer object


    end

    methods

        function self = autoencoderModel( lossFcns, ...
                                          superArgs, ...
                                          args )
            % Initialize the model
            arguments (Repeating)
                lossFcns      lossFunction
            end
            arguments
                superArgs.?representationModel
                args.XOutDim        double = 0
                args.hasSeqInput    logical = false
                args.isVAE          logical = false
                args.weights        double ...
                                    {mustBeNumeric,mustBeVector} = 1
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@representationModel( superArgsCell{:} );

            % placeholders for subclasses to define
            self.nets.encoder = [];
            self.nets.decoder = [];
            self.netNames = {'encoder', 'decoder'};
            self.nNets = 2;
            self.isVAE = args.isVAE;
            self.hasSeqInput = args.hasSeqInput;

            if args.XOutDim == 0
                self.XOutDim = self.XDim;
            end

            % copy over the loss functions associated
            % and any networks with them for later training 
            self = addLossFcns( self, lossFcns{:}, weights = args.weights );

            % check a reconstruction loss is present
            if ~any( self.lossFcnTbl.types=='Reconstruction' )
                eid = 'aeModel:NoReconstructionLoss';
                msg = 'No reconstruction loss object has been specified.';
                throwAsCaller( MException(eid,msg) );
            end 

            self.nLoss = sum( self.lossFcnTbl.nLosses );

            % indicate that further initialization is required
            self.isInitialized = false;

        end


        function self = addLossFcns( self, newFcns, args )
            % Add one or more loss function objects to the model
            arguments
                self
            end
            arguments (Repeating)
                newFcns   lossFunction
            end
            arguments
                args.weights double {mustBeNumeric,mustBeVector} = 1
            end
       
            nFcns = length( newFcns );

            % check the weights
            if args.weights==1
                % default is to assign a weight of 1 to all functions
                w = ones( nFcns, 1 );
            elseif length( args.weights ) ~= nFcns
                % weights don't correspond to the functions
                eid = 'aeModel:WeightsMismatch';
                msg = 'Number of assigned weights does not match number of functions';
                throwAsCaller( MException(eid,msg) );
            else
                w = args.weights;
            end
            self.lossFcnWeights = [ self.lossFcnWeights w ];

            % update the names list
            self.lossFcnNames = [ self.lossFcnNames getFcnNames(newFcns) ];
            % add to the loss functions
            for i = 1:length( newFcns )
                self.lossFcns.(newFcns{i}.name) = newFcns{i};
            end

            % add networks, if required
            self = addLossFcnNetworks( self, newFcns );

            % store the loss functions' details 
            % and relevant details for easier access when training
            self = self.setLossInfoTbl;
            self.lossFcnTbl.types = categorical( self.lossFcnTbl.types );
            self.lossFcnTbl.inputs = categorical( self.lossFcnTbl.inputs );

        end


        function self = addLossFcnNetworks( self, newFcns )
            % Add one or more networks to the model
            arguments
                self
                newFcns
            end

            nFcns = length( newFcns );
            k = length( self.nets );
            for i = 1:nFcns
                thisLossFcn = newFcns{i};
                if thisLossFcn.hasNetwork
                    k = k+1;
                    % add the network object
                    self.nets.(thisLossFcn.name) = thisLossFcn.initNetwork;
                    % record its name
                    self.netNames = [ string(self.netNames) thisLossFcn.name ];
                    % increment the counter
                    self.nNets = self.nNets + 1;
                end
            end


        end


        function self = initTrainer( self, args )
            arguments
                self
                args.?modelTrainer
            end

            % set the trainer's properties
            argsCell = namedargs2cell( args );
            self.trainer = modelTrainer( argsCell{:} );

            % confirm if initialization is complete
            self.isInitialized = ~isempty(self.optimizer);

        end
        

        function self = initOptimizer( self, args )
            arguments
                self
                args.?modelOptimizer
            end

            % set the trainer's properties
            argsCell = namedargs2cell( args );
            self.optimizer = modelOptimizer( self.netNames, argsCell{:} );

            % confirm if initialization is complete
            self.isInitialized = ~isempty(self.trainer);

        end


        function [ self, lossTrn, lossVal ] = train( self, thisDataset )
            % Train the autoencoder model
            arguments
                self          autoencoderModel
                thisDataset   modelDataset
            end

            if ~self.isInitialized
                eid = 'Autoencoder:NotInitialized';
                msg = 'The trainer, optimizer or dataset parameters have not been set.';
                throwAsCaller( MException(eid,msg) );
            end

            % check dataset is suitable
            if thisDataset.isFixedLength == self.hasSeqInput
                eid = 'aeModel:DatasetNotSuitable';
                if thisDataset.isFixedLength
                    msg = 'The dataset should have variable length for the model.';
                else
                    msg = 'The dataset should have fixed length for the model.';
                end
                throwAsCaller( MException(eid,msg) );
            end

            % re-partition the data to create training and validation sets
            cvPart = cvpartition( thisDataset.nObs, 'Holdout', 0.25 );
            
            thisTrnSet = thisDataset.partition( training(cvPart) );
            thisValSet = thisDataset.partition( test(cvPart) );

            % run the training loop
            [ self, self.optimizer, lossTrn, lossVal ] = ...
                            self.trainer.runTraining( self, ...
                                                      thisTrnSet, ...
                                                      thisValSet );


        end

    end


    methods (Access = protected)

        function self = setLossInfoTbl( self )
            % Update the info table
            
            nFcns = length( self.lossFcnNames );
            names = strings( nFcns, 1 );
            types = strings( nFcns, 1 );
            inputs = strings( nFcns, 1 );
            weights = zeros( nFcns, 1 );
            nLosses = zeros( nFcns, 1 );
            lossNets = strings( nFcns, 1 );
            hasNetwork = false( nFcns, 1 );
            doCalcLoss = false( nFcns, 1 );
            useLoss = false( nFcns, 1 );

            for i = 1:nFcns
                
                thisLossFcn = self.lossFcns.(self.lossFcnNames(i));
                names(i) = thisLossFcn.name;
                types(i) = thisLossFcn.type;
                inputs(i) = thisLossFcn.input;
                weights(i) = self.lossFcnWeights(i);
                nLosses(i) = thisLossFcn.nLoss;
                hasNetwork(i) = thisLossFcn.hasNetwork;
                doCalcLoss(i) = thisLossFcn.doCalcLoss;
                useLoss(i) = thisLossFcn.useLoss;

                nNets = length(thisLossFcn.lossNets);
                for j = 1:nNets
                    if length(string( thisLossFcn.lossNets{j} ))==1
                        assignments = thisLossFcn.lossNets{j};
                    else
                        assignments = strjoin( thisLossFcn.lossNets{j,:}, '+' );
                    end
                    lossNets(i) = strcat( lossNets(i), assignments ) ;
                    if j<nNets
                        lossNets(i) = strcat( lossNets(i), "; " );
                    end
                end

            end

            self.lossFcnTbl = table( names, types, inputs, weights, ...
                    nLosses, lossNets, hasNetwork, doCalcLoss, useLoss );

        end

    end


    methods (Static)

        function Z = encode( self, dlX )
            % Encode features Z from X using the model

            Z = predict( self.nets.encoder, dlX );

        end

        function XHat = reconstruct( self, Z )
            % Reconstruct X from Z using the model

            XHat = predict( self.nets.decoder, Z );


        end

        function net = getNetwork( self, name )
            arguments
                self
                name         string {mustBeNetName( self, name )}
            end
            
            net = self.nets.(name);
            if isa( net, 'lossFunction' )
                net = self.lossFcns.(name).net;
            end

        end

        
        function names = getNetworkNames( self )
            arguments
                self
            end
            
            names = fieldnames( self.nets );

        end


        function loss = getReconLoss( self, dlX, dlXHat )
            % Calculate the reconstruction loss
            arguments
                self
                dlX     dlarray
                dlXHat  dlarray
            end
            
            name = self.lossFcnTbl.names( self.lossFcnTbl.types=='Reconstruction' );
            loss = self.lossFcns.(name).calcLoss( dlX, dlXHat );

        end


        function isValid = mustBeNetName( self, name )
            arguments
                self
                name
            end

            isValid = ismember( name, self.names );

        end


        function dlXC = latentComponents( decoder, dlZ, args )
            % Calculate the funtional components from the latent codes
            % using the decoder network. For each component, the relevant 
            % code is varied randomly about the mean. This is more 
            % efficient than calculating two components at strict
            % 2SD separation from the mean.
            arguments
                decoder         dlnetwork
                dlZ             dlarray
                args.nSample    double {mustBeInteger,mustBePositive} = 10
                args.dlZMean    dlarray = []
                args.dlZLogVar  dlarray = []
            end

            nSample = args.nSample;
            ZDim = size( dlZ, 1 );
            if isempty( args.dlZMean ) || isempty( args.dlZLogVar )
                % compute the mean and SD across the batch
                dlZMean = mean( dlZ, 2 );
                dlZStd = std( dlZ, [], 2 );
            else
                % use the assigned mean and SD for the batch
                dlZMean = args.dlZMean;
                dlZStd = sqrt(exp( args.dlZLogVar ));
            end
            
            % initialise the components' Z codes at the mean
            % include an extra one that will be preserved
            dlZC = repmat( dlZMean, 1, ZDim*nSample+1 );

            % convert to double to greatly speed up processing
            % (tracing is not an issue)
            ZC = double(extractdata( dlZC ));
            ZMean = double(extractdata( dlZMean ));
            ZStd = double(extractdata( dlZStd ));
            
            for j = 1:nSample
                offset = 2*randn;
                for i =1:ZDim
                    % vary ith component randomly about its mean
                    ZC(i,(i-1)*nSample+j) = ZMean(i) + offset*ZStd(i);
                end
            end

            % convert back
            dlZC = dlarray( ZC, 'CB' );
           
            % generate all the component curves using the decoder
            dlXC = forward( decoder, dlZC );

            % centre about the mean curve (last curve)
            XDim = size( dlXC );
            if length( XDim )==2
                dlXC = dlXC( :, 1:end-1 ) - dlXC( :, end );
            else
                dlXC = dlXC( :, :, 1:end-1 ) - dlXC( :, :, end );
            end
            
        end
            

        function plotLatentComp( self, Z, c, tSpan, fdPar )
            % Plot characteristic curves of the latent codings which are similar
            % in conception to the functional principal components
            
            zScores = linspace( -2, 2, 3 );
            
            % find the mean latent scores for centring 
            ZMean = repmat( mean( Z, 2 ), 1, length(zScores) );
            
            nPlots = min( self.ZDim, length(ax) );           
            for i = 1:nPlots
                
                % initialise component codes at their mean values
                ZC = ZMean;
            
                % vary code i about its mean in a standardised way
                Zsd = std( Z(i,:) );
                for j = 1:length(zScores)
                    ZC(i,j) = ZMean(i,1) + zScores(j)*Zsd;
                end
                
                % duplicate for each class
                dlZC = dlarray( ZC, 'CB' );
            
                % generate the curves using the decoder
                dlXC = reconstruct( self, dlZC );
                XC = double( extractdata( dlXC ) );
            
                if size(XC,3) > 1
                    % select the requested channel
                    XC = squeeze( XC(:,c,:) );
                end
                
                % convert into smooth function
                XCFd = smooth_basis( tSpan, XC, fdPar );
            
                % plot the curves
                subplotFd( ax(i), XCFd );
            
            end
        
        
        end
    
    
    
    end

end


function names = getFcnNames( lossFcns )

    nFcns = length( lossFcns );
    names = strings( nFcns, 1 );
    for i = 1:nFcns
        names(i) = lossFcns{i}.name;
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





