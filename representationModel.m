classdef representationModel
    % Super class encompassing dimensional reduction models

    properties
        nInputs    % X dimension (number of points)
        nFeatures  % Z dimension (number of features)
        nChannels  % number of channels in X
        encoder    % encoding function parameters
        decoder    % decoding function parameters

    end

    methods
        function obj = representationModel( nameValueArgs )
            % Initialize the model
            arguments
                nameValueArgs.nInputs    single {mustBeInteger, mustBePositive} = 10
                nameValueArgs.nFeatures  single {mustBeInteger, mustBePositive} = 1
                nameValueArgs.nChannels  single {mustBeInteger, mustBePositive} = 1
            end
            obj.nInputs = nameValueArgs.nInputs;
            obj.nFeatures = nameValueArgs.nFeatures;
            obj.nChannels = nameValueArgs.nChannels;
            obj.encoder = [];
            obj.decoder = [];
        end

        function obj = train( obj, X )
            % Train the model - placeholder
            obj.encoder = std(X);
            obj.decoder = mean(X);

        end

        function Z = encode( obj, X )
            % Encode features Z from X using the model - placeholder
            nRows = size( X, 2 );
            xi = 1:obj.nInputs;
            zi = linspace( 1, obj.nInputs, obj.nFeatures );
            coding = interp1( xi, obj.encoder, zi, 'linear' );
            Z = randn( nRows, 1 )*coding;
        end

        function XHat = reconstruct( obj, Z )
            % Reconstruct X from Z using the model - placeholder
            zi = 1:obj.nFeatures;
            xi = linspace( 1, obj.nFeatures, obj.nInputs );
            coding = interp1( zi, Z', xi, 'linear' );
            XHat = obj.decoder + coding;
        end

        function err = loss( obj, X, XHat )
            % Compute the  - placeholder
            err = mse( X, XHat );
        end


    end
end