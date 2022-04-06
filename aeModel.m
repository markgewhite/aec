% ************************************************************************
% Class: aeModel
%
% Subclass defining the framework for an autoencoder model
%
% ************************************************************************

classdef aeModel < representationModel

    properties
        nets         % networks defined in this model (structure)
        lossFcns     % loss functions
        isVAE        % flag indicating if variational autoencoder
    end

    methods

        function self = aeModel( superArgs, args )
            % Initialize the model
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
            self.lossFcns.recon = @reconLoss;
            self.isVAE = args.isVAE;

        end

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
                                                  dlXReal, dlXNReal, ... 
                                                  dlYReal, ...
                                                  doTrainAE )
            arguments
                self
                dlXReal      dlarray  % input to the encoder
                dlXNReal     dlarray  % output target for the decoder
                dlYReal      dlarray  % auxiliary outcome variable
                doTrainAE    logical  % whether to train the AE
            end

            batchSize = size( dlXReal, 2 );
            if self.isVAE
                % duplicate X & C to reflect mulitple draws of VAE
                dlXReal = repmat( dlXReal, 1, self.nets.encoder.nDraws );
                dlYReal = repmat( dlYReal, self.nets.encoder.nDraws, 1 );
            end
            
            if doTrainAE
                % autoencoder training
            
                % generate latent encodings
                [ dlZFake, state.encoder ] = forward( self.nets.encoder, dlXReal);
    
                % reconstruct curves from latent codes
                [ dlXFake, state.decoder ] = forward( self.nets.decoder, dlZFake );
                
                % calculate the reconstruction loss
                loss.recon = squeeze(mean( (dlXFake - dlXReal).^2, 'all' ));

                if self.adversarial   
                    % predict authenticity from real Z using the discriminator
                    dlZReal = dlarray( randn( self.ZDim, batchSize ), 'CB' );
                    dlDReal = forward( self.nets.discriminator, dlZReal );
                    
                    % predict authenticity from fake Z
                    [ dlDFake, state.discriminator ] = ...
                                forward( self.nets.discriminator, dlZFake );
                    
                    % discriminator loss for Z
                    loss.dis = -setup.reg.dis* ...
                                    0.5*mean( log(dlDReal + eps) + log(1 - dlDFake + eps) );
                    loss.gen = -setup.reg.gen* ...
                                    mean( log(dlDFake + eps) );
                    
                else
                    loss.dis = 0;
                    loss.gen = 0;
                    state.dis = [];
                end
                
                if setup.variational && ~setup.adversarial && ~setup.wasserstein
                    % calculate the variational loss
                    loss.var = -setup.reg.beta* ...
                        0.5*mean( sum(1 + dlLogVar - dlMu.^2 - exp(dlLogVar)) );
                else
                    loss.var = 0;
                end
    
                if setup.wasserstein
                    % calculate the maximum mean discrepancy loss
                    dlZReal = dlarray( randn( setup.zDim, setup.batchSize ), 'CB' );
                    loss.mmd = mmdLoss( dlZFake, dlZReal, setup.mmd );
                else
                    loss.mmd = 0;
                end

            else
                % no autoencoder training
                dlZFake = predict( self.nets.encoder, dlXReal );
                                
                loss.recon = 0;
                loss.dis = 0;
                loss.gen = 0;
                loss.var = 0;
                loss.mmd = 0;
                state.enc = [];
                state.dec = [];
                state.dis = [];
            
            end




        end

    end

end



