% ************************************************************************
% Class: aeModel
%
% Subclass defining the framework for an autoencoder model
%
% ************************************************************************

classdef aeModel < representationModel

    properties
        nets         % networks defined in this model (structure)
        netNames     % names of the networks (for convenience)
        lossFcns     % array of loss functions
        lossFcnTbl   % table loss function details
        isVAE        % flag indicating if variational autoencoder
    end

    methods

        function self = aeModel( lossFcns, superArgs, args )
            % Initialize the model
            arguments (Repeating)
                lossFcns     lossFunction   
            end
            arguments
                superArgs.?representationModel
                args.isVAE  logical = false
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@representationModel( superArgsCell{:} );

            % placeholders for subclasses to define
            self.nets.encoder = [];
            self.nets.decoder = [];
            self.netNames = {'encoder','decoder'};
            self.isVAE = args.isVAE;

            % copy any networks associated with the loss functions
            % into this object for later training 
            self = addLossFcns( self, lossFcns );

            % store the loss functions' details 
            % and relevant details for easier access when training
            self.lossFcnTbl = self.lossInfoTbl;
            self.lossFcnTbl.types = categorical( self.lossFcnTbl.types );
            self.lossFcnTbl.inputs = categorical( self.lossFcnTbl.inputs );

            % check a reconstruction loss is present
            if ~any( self.lossFcnTbl.types=='Reconstruction' )
                eid = 'aeModel:NoReconstructionLoss';
                msg = 'No reconstruction loss object has been specified.';
                throwAsCaller( MException(eid,msg) );
            end 



        end


        function self = addNetwork( self, newFcns )
            % Add one or more networks to the model
            arguments
                self
                newFcns
            end

            nFcns = length( newFcns );
            for i = 1:nFcns
                if newFcns{i}.hasNetwork
                    name = newFcns{i}.name;
                    % add the network object
                    self.nets.(name) = newFcns{i}.net;
                    % record its name
                    self.netNames = [ self.netNames name ];
                end
            end

        end


        function self = addLossFcns( self, newFcns )
            % Add one or more loss function objects to the model
            arguments
                self
                newFcns
            end

            % add loss function objects and update the info table
            self = addToLossFcnObjs( self, newFcns );
            self = addToLossFcnTbl( self, newFcns );

        end




    end


    methods (Access = protected)

        function self = addToLossFcnObjs( self, newFcns )
            % Add one or more loss function objects to the model
            arguments
                self
                newFcns
            end

            newNames = self.getFcnNames( newFcns );
            % add to loss functions structure
            for i = 1:length( newNames )
                self.lossFcns.(newNames(i)) = newFcns{i};
            end

        end


        function self = addToLossFcnTbl( self, newFcns )
            % Add one or more loss functions to the model
            arguments
                self
                newFcns
            end

            % update the info table
            newFcnTbl = self.lossInfoTbl( newFcns );
            self.lossFcnTbl = [ self.lossFcnTbl; newFcnTbl ];
            self.lossFcnTbl.types = categorical( self.lossFcnTbl.types );
            self.lossFcnTbl.inputs = categorical( self.lossFcnTbl.inputs );

        end


        function T = lossInfoTbl( self )
            arguments
                self
            end

            theseLossFcns = self.lossFcns;

            % update the info table
            nFcns = length( theseLossFcns );
            names = strings( nFcns, 1 );
            types = strings( nFcns, 1 );
            inputs = strings( nFcns, 1 );
            hasState = false( nFcns, 1 );
            doCalcLoss = false( nFcns, 1 );
            useLoss = false( nFcns, 1 );
            for i = 1:nFcns
                
                names(i) = theseLossFcns.name;
                types(i) = theseLossFcns.type;
                inputs(i) = theseLossFcns.input;
                hasState(i) = theseLossFcns.hasNetwork;
                doCalcLoss(i) = theseLossFcns.doCalcLoss;
                useLoss(i) = theseLossFcns.useLoss;

            end

            T = table( names, types, inputs, ...
                        hasState, doCalcLoss, useLoss );

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

        function names = getFcnNames( self, lossFcns )

            % how to call a routine without self being required??

            nFcns = length( lossFcns );
            names = strings( nFcns, 1 );
            for i = 1:nFcns
                names(i) = lossFcns.name;
            end

        end



        function [grad, state, loss] = gradients( self, ...
                                                  dlXIn, dlXOut, ... 
                                                  dlY, ...
                                                  doTrainAE )
            arguments
                self
                dlXIn        dlarray  % input to the encoder
                dlXOut       dlarray  % output target for the decoder
                dlY          dlarray  % auxiliary outcome variable
                doTrainAE    logical  % whether to train the AE
            end

            thisEncoder = self.nets.encoder;
            thisDecoder = self.nets.decoder;

            if self.isVAE
                % duplicate X & C to reflect mulitple draws of VAE
                dlXOut = repmat( dlXOut, 1, thisEncoder.nDraws );
                dlY = repmat( dlY, thisEncoder.nDraws, 1 );
            end
            
            if doTrainAE
                % autoencoder training
            
                % generate latent encodings
                [ dlZGen, state.encoder ] = forward( thisEncoder, dlXIn);
    
                % reconstruct curves from latent codes
                [ dlXGen, state.decoder ] = forward( thisDecoder, dlZGen );
                
            else
                % no autoencoder training
                dlZGen = predict( thisEncoder, dlXIn );
            
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

            % initialize the loss accumulator by network
            lossAccum = table2struct( table( ...
                    'Size', [1, length(self.netNames)], ...
                    'VariableNames', self.netNames, ...
                    'VariableTypes', ...
                            repmat("double", [1, length(self.netNames)]) ));
            
            nFcns = size( activeFcns, 1 );
            nLoss = sum( activeFcns.nLoss );
            loss = zeros( nLoss, 1 );
            j = 1;
            for i = 1:nFcns
               
                % identify the loss function
                thisName = activeFcns.names(i);
                % take the model's copy of the loss function object
                thisLossFcn = self.lossFcns.name;

                % assign indices for the number of losses returned
                lossIdx = j:j+thisLossFcn.nLoss(i)-1;
                j = j + thisLossFcn.nLoss;

                % select the input variables
                switch thisLossFcn.inputs
                    case 'X-XHat'
                        dlV = { dlXOut, dlXGen };
                    case 'XC'
                        dlV = dlXC;
                    case 'Z'
                        dlV = dlZGen;
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
                        [ loss( lossIdx ), state.(thisName) ] = ...
                            thisLossFcn.doCalcLoss( ...
                                self.nets.(thisName), dlV{:} );
                    else
                        loss( lossIdx ) = thisLossFcn.calcLoss( ...
                                self.nets.(thisName), dlV{:} );
                    end
                else
                    % call the loss function straightforwardly
                    loss( lossIdx ) = thisLossFcn.calcLoss( dlV{:} );
                end

                % assign loss to loss accumulator for associated network(s)
                for j = 1:length( lossIdx )
                    for k = 1:length( thisLossFcn.lossNets(j,:) )
                        netName = thisLossFcn.lossNets(i,j);
                        lossAccum.(netName) = ...
                            lossAccum.(netName) + loss( lossIdx(j) );
                    end
                end

            end

        % compute the gradients for each network
        for i = 1:length(self.netNames)

            thisName = self.netNames(i);
            thisNetwork = self.nets.(thisName);
            grad.(thisName) = dlgradient( thisNetwork, ...
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



