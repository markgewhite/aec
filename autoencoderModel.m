% ************************************************************************
% Class: autoencoderModel
%
% Subclass defining the framework for an autoencoder model
%
% ************************************************************************

classdef autoencoderModel < representationModel

    properties
        nets           % networks defined in this model (structure)
        netNames       % names of the networks (for convenience)
        isVAE          % flag indicating if variational autoencoder
        lossFcns       % array of loss functions
        lossFcnNames   % names of the loss functions
        lossFcnWeights % weights to be applied to the loss function
        lossFcnTbl     % convenient table summarising loss function details
    end

    methods

        function self = autoencoderModel( lossFcns, superArgs, args )
            % Initialize the model
            arguments (Repeating)
                lossFcns      lossFunction
            end
            arguments
                superArgs.?representationModel
                args.isVAE    logical = false
                args.weights  double {mustBeNumeric,mustBeVector} = 1
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@representationModel( superArgsCell{:} );

            % placeholders for subclasses to define
            self.nets.encoder = [];
            self.nets.decoder = [];
            self.netNames = {'encoder', 'decoder'};
            self.isVAE = args.isVAE;

            % copy over the loss functions associated
            % and any networks with them for later training 
            self = addLossFcns( self, lossFcns{:}, weights = args.weights );

            % check a reconstruction loss is present
            if ~any( self.lossFcnTbl.types=='Reconstruction' )
                eid = 'aeModel:NoReconstructionLoss';
                msg = 'No reconstruction loss object has been specified.';
                throwAsCaller( MException(eid,msg) );
            end 



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
                    self.nets.(thisLossFcn.name) = thisLossFcn;
                    % record its name
                    self.netNames = [ string(self.netNames) thisLossFcn.name ];
                end
            end

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
                    lossNets(i) = strcat( lossNets(i), ...
                                          thisLossFcn.lossNets(j) ) ;
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

        function vae = makeVAE( net )
            % Convert an encoder into a variational encoder
            arguments
                net  dlnetwork
            end

            vae = vaeNetwork( net );

        end

        function Z = encode( self, X )
            % Encode features Z from X using the model

            Z = predict( self.nets.encoder, X );

        end

        function XHat = reconstruct( self, Z )
            % Reconstruct X from Z using the model

            XHat = predict( self.nets.decoder, Z );


        end

        function net = getNetwork( self, name )
            arguments
                self
                name         string ...
                    {mustBeNetName( self, name )}
            end
            
            net = self.nets.(name);
            if isa( net, 'lossFunction' )
                net = self.lossFcns.(name).net;
            end

        end


        function isValid = mustBeNetName( self, name )
            arguments
                self
                name
            end

            isValid = ismember( name, self.names );

        end



        function [grad, state, loss] = gradients( self, ...
                                                  nets, ...
                                                  dlXIn, dlXOut, ... 
                                                  dlY, ...
                                                  doTrainAE )
            arguments
                self
                nets         struct   % networks, made explicit for tracing
                dlXIn        dlarray  % input to the encoder
                dlXOut       dlarray  % output target for the decoder
                dlY          dlarray  % auxiliary outcome variable
                doTrainAE    logical  % whether to train the AE
            end

            if self.isVAE
                % duplicate X & C to reflect mulitple draws of VAE
                dlXOut = repmat( dlXOut, 1, thisEncoder.nDraws );
                dlY = repmat( dlY, thisEncoder.nDraws, 1 );
            end
            
            if doTrainAE
                % autoencoder training
            
                % generate latent encodings
                [ dlZGen, state.encoder ] = forward( nets.encoder, dlXIn);
    
                % reconstruct curves from latent codes
                [ dlXGen, state.decoder ] = forward( nets.decoder, dlZGen );
                
            else
                % no autoencoder training
                dlZGen = predict( nets.encoder, dlXIn );
            
            end


            % select the active loss functions
            activeFcns = self.lossFcnTbl( self.lossFcnTbl.doCalcLoss, : );

            if any( activeFcns.types=='Component' )
                % compute the AE components
                if self.isVAE
                    dlXC = latentComponents( self, dlZGen, ...
                                    dlZMean = self.dlZMeans, ...
                                    dlZLogVar = self.dlZLogVars );
                else
                    dlXC = latentComponents( self, dlZGen );
                end
            end

            
            % compute the active loss functions in turn
            % and assign to networks
            
            nFcns = size( activeFcns, 1 );
            nLoss = sum( activeFcns.nLosses );
            loss = zeros( nLoss, 1 );
            idx = 1;
            for i = 1:nFcns
               
                % identify the loss function
                thisName = activeFcns.names(i);
                % take the model's copy of the loss function object
                thisLossFcn = self.lossFcns.(thisName);

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
                end

                % calculate the loss
                % (make sure to use the model's copy 
                %  of the relevant network object)
                if thisLossFcn.hasNetwork
                    % call the loss function with the network object
                    if thisLossFcn.hasState
                        % and store the network state too
                        [ thisLossFcn, thisLoss, state.(thisName) ] = ...
                            thisLossFcn.calcLoss( dlV{:} );
                    else
                        thisLoss = thisLossFcn.calcLoss( dlV{:} );
                    end
                else
                    % call the loss function straightforwardly
                    thisLoss = thisLossFcn.calcLoss( dlV{:} );
                end
                loss( lossIdx ) = thisLoss;

                % assign loss to loss accumulator for associated network(s)
                for j = 1:length( lossIdx )
                    for k = 1:length( thisLossFcn.lossNets(j,:) )
                        netName = thisLossFcn.lossNets(j,k);
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

        % compute the gradients for each network
        for i = 1:length(self.netNames)

            thisName = self.netNames{i};
            thisNetwork = nets.(thisName);
            grad.(thisName) = dlgradient( lossAccum.(thisName), ...
                                          thisNetwork.Learnables, ...
                                          'RetainData', true );
            
        end

        end


        function dlXC = latentComponents( self, dlZ, args )
            % Calculate the funtional components from the latent codes
            % using the decoder network. For each component, the relevant 
            % code is varied randomly about the mean. This is more 
            % efficient than calculating two components at strict
            % 2SD separation from the mean.
            arguments
                self
                dlZ             dlarray
                args.nSample    double {mustBeInteger,mustBePositive} = 10
                args.dlZMean    dlarray = []
                args.dlZLogVar  dlarray = []
            end

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
            dlZC = repmat( dlZMean, 1, self.ZDim*nSample+1 );
            
            for i =1:self.ZDim
                for j = 1:nSample
                
                    % vary ith component randomly about its mean
                    dlZC(i,(i-1)*nSample+j) = dlZMean(i) + 2*randn*dlZStd(i);
                    
                end
            end
            
            % generate all the component curves using the decoder
            dlXC = forward( self.nets.decoder, dlZC );
            
            % centre about the mean curve (last curve)
            dlXC = dlXC(:,1:end-1) - dlXC(:,end);
            
        end
            

        function plotLatentComp( ax, self, Z, c, tSpan, fdPar )
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


