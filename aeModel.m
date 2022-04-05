classdef aeModel < representationModel
    % Class defining an autoencoder model

    properties
        nets         % networks defined in this model (structure)
        lossFcns     % loss functions
    end

    methods

        function self = aeModel( superArgs, args )
            % Initialize the model
            arguments
                superArgs.?representationModel
            end

            % set the superclass's properties
            superArgsCell = namedargs2cell( superArgs );
            self = self@representationModel( superArgsCell{:} );

            % placeholders for subclasses to define
            self.nets.encoder = [];
            self.nets.decoder = [];
            self.lossFcns.recon = @reconLoss;

        end

        function model = train( model, trainer, X, XN )
            arguments
                model
                trainer {mustBeA(trainer, 'trainer')}
                X
                XN
            end

            model = train( trainer, model, X, XN ); 

        end

        function Z = encode( self, X )
            % Encode features Z from X using the model

            Z = predict( self.nets.encoder, X );

        end

        function XHat = reconstruct( self, Z )
            % Reconstruct X from Z using the model

            XHat = predict( self.nets.decoder, Z );


        end




    end

end



