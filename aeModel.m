% ************************************************************************
% Class: aeModel
%
% Subclass defining the framework for an autoencoder model
%
% ************************************************************************

classdef aeModel < representationModel

    properties
        nets         % networks defined in this model (structure)
        lossFcns     % array of loss functions
        lossFcnTbl   % table loss function details
        isVAE        % flag indicating if variational autoencoder
    end

    methods

        function self = aeModel( lossFcn, superArgs, args )
            % Initialize the model
            arguments (Repeating)
                lossFcn     lossFunction   
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
            self.isVAE = args.isVAE;

            % store the loss functions 
            % and relevant details for easier access when training
            self.lossFcns = lossFcn;
            nFcns = length( lossFcn );
            names = strings( nFcns, 1 );
            types = strings( nFcns, 1 );
            inputs = strings( nFcns, 1 );
            hasState = false( nFcns, 1 );
            doCalcLoss = false( nFcns, 1 );
            useLoss = false( nFcns, 1 );
            hasReconLoss = false;
            for i = 1:nFcns
                
                names(i) = self.lossFcns{i}.name;
                types(i) = self.lossFcns{i}.type;
                inputs(i) = self.lossFcns{i}.input;
                hasState(i) = self.lossFcns{i}.hasNetwork;
                doCalcLoss(i) = self.lossFcns{i}.doCalcLoss;
                useLoss(i) = self.lossFcns{i}.useLoss;

                hasReconLoss = hasReconLoss || ...
                    strcmpi( lossFcn{i}.type, 'Reconstruction' );
                if lossFcn{i}.hasNetwork
                    self.nets.(lossFcn{i}.name) = lossFcn{i}.net;
                end

            end

            if ~hasReconLoss
                eid = 'aeModel:NoReconstructionLoss';
                msg = 'No reconstruction loss object has been specified.';
                throwAsCaller( MException(eid,msg) );
            end

            self.lossFcnTbl = table( names, types, inputs, ...
                                        hasState, doCalcLoss, useLoss );
            self.lossFcnTbl.types = categorical( self.lossFcnTbl.types );
            self.lossFcnTbl.inputs = categorical( self.lossFcnTbl.inputs );

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

            if self.isVAE
                % duplicate X & C to reflect mulitple draws of VAE
                dlXOut = repmat( dlXOut, 1, self.nets.encoder.nDraws );
                dlY = repmat( dlY, self.nets.encoder.nDraws, 1 );
            end
            
            if doTrainAE
                % autoencoder training
            
                % generate latent encodings
                [ dlZGen, state.encoder ] = forward( self.nets.encoder, dlXIn);
    
                % reconstruct curves from latent codes
                [ dlXGen, state.decoder ] = forward( self.nets.decoder, dlZGen );
                
            else
                % no autoencoder training
                dlZGen = predict( self.nets.encoder, dlXIn );
            
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
            nFcns = size( activeFcns, 1 );
            loss = zeros( nFcns, 1 );
            for i = 1:nFcns
               
                name = activeFcns.names(i);
                % select the input variables
                switch activeFcns.inputs(i)
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
                if activeFcns.hasState(i)
                    % and store the network state too
                    [ loss(i), state.(name) ] = ...
                        self.lossFcns{i}.doCalcLoss( ...
                            self.nets.(name), dlV{:} );
                else
                    loss(i) = self.lossFcns{i}.calcLoss( ...
                            self.nets.(name), dlV{:} );
                end

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



