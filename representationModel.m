classdef representationModel
    % Super class encompassing dimensional reduction models

    properties
        nInputs    % X dimension (number of points)
        nFeatures  % Z dimension (number of features)
        nChannels  % number of channels in X
    end

    methods
        function obj = representationModel( args )
            % Initialize the model
            arguments
                args.nInputs    single {mustBeInteger, mustBePositive} = 10
                args.nFeatures  single {mustBeInteger, mustBePositive} = 1
                args.nChannels  single {mustBeInteger, mustBePositive} = 1
            end
            obj.nInputs = args.nInputs;
            obj.nFeatures = args.nFeatures;
            obj.nChannels = args.nChannels;
        end

        function obj = train( obj, X )
            % Train the model - placeholder
        end

        function Z = encode( obj, X )
            % Encode features Z from X using the model - placeholder
        end

        function XHat = reconstruct( obj, Z )
            % Reconstruct X from Z using the model - placeholder
        end

        function err = loss( obj, X, XHat )
            % Compute the  - placeholder
            err = mse( X, XHat );
        end


    end
end