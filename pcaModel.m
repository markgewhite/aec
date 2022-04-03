classdef pcaModel < representationModel
    % Class defining a PCA model

    properties
        tSpan     double  % timespan vector for FDA
        basisFd      % functional data basis
        fdParams     % functional data parameters
        varProp   double  % explained variance
    end

    methods

        function obj = pcaModel( nameValueArgs )
            % Initialize the model
            arguments
                nameValueArgs.?representationModel
                nameValueArgs.tSpan    double {mustBeVector}
                nameValueArgs.basisFd
                nameValueArgs.fdParams
            end

            f = nameValueArgs.nFeatures;

            obj = obj@representationModel( 'nFeatures', f );

            obj.tSpan = nameValueArgs.tSpan;
            obj.basisFd = nameValueArgs.basisFd;
            obj.fdParams = nameValueArgs.fdParams;

        end


        function obj = train( obj, XFd )
            % Run FPCA for the encoder
            pcaStruct = pca_fd( XFd, obj.nFeatures );
            obj.encoder = pcaStruct.fdhatfd;
            obj.decoder = pcaStruct.meanfd;
            obj.varProp = pcaStruct.varProp;

        end

        function Z = encode( obj, XFd )
            % Encode features Z from X using the model
            nRows = size( X, 2 );

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