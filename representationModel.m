classdef representationModel
    % Super class encompassing dimensional reduction models

    properties
        XDim       % X dimension (number of points)
        ZDim       % Z dimension (number of features)
        XChannels  % number of channels in X
    end

    methods
        function obj = representationModel( args )
            % Initialize the model
            arguments
                args.XDim    single {mustBeInteger, mustBePositive} = 10
                args.ZDim  single {mustBeInteger, mustBePositive} = 1
                args.XChannels  single {mustBeInteger, mustBePositive} = 1
            end
            obj.XDim = args.XDim;
            obj.ZDim = args.ZDim;
            obj.XChannels = args.XChannels;
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

        function err = reconLoss( obj, X, XHat )
            % Compute the  - placeholder
            err = mse( X, XHat );
        end


    end
end