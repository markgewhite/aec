classdef pcaModel < representationModel
    % Class defining a PCA model

    properties
        meanFd                % mean curve
        compFd                % functional principal components
        varProp       double  % explained variance
        basisFd               % functional basis
        tSpan         double  % time span
        nKnots        double  % number of knots
        lambda        double  % roughness penalty
        penaltyOrder  double  % roughness penalty order
    end

    methods

        function obj = pcaModel( fdParameters, penaltyOrder, superArgs )
            % Initialize the model
            arguments
                fdParameters  % functional data parameters object
                penaltyOrder  % roughness penalty order 
                superArgs.?representationModel
            end

            argsCell = namedargs2cell(superArgs);
            obj = obj@representationModel( argsCell{:} );

            obj.meanFd = [];
            obj.compFd = [];
            obj.varProp = [];

            obj.basisFd = getbasis( fdParameters );
            obj.tSpan = getbasispar( obj.basisFd );
            obj.nKnots = getnbasis( obj.basisFd );
            obj.lambda = getlambda( fdParameters );
            obj.penaltyOrder = penaltyOrder;

        end


        function obj = train( obj, XFd )
            % Run FPCA for the encoder
            arguments
                obj
                XFd {mustBeValidBasis(obj, XFd)}  
            end

            pcaStruct = pca_fd( XFd, obj.nFeatures );

            obj.meanFd = pcaStruct.meanfd;
            obj.compFd = pcaStruct.harmfd;
            obj.varProp = pcaStruct.varprop;

        end

        function Z = encode( obj, XFd )
            % Encode features Z from X using the model
            arguments
                obj
                XFd {mustBeValidBasis(obj, XFd)}  
            end

            Z = pca_fd_score( XFd, obj.meanFd, obj.compFd, ...
                              obj.nFeatures, true );

        end

        function XHatFd = reconstruct( obj, Z )
            % Reconstruct X from Z using the model
            arguments
                obj
                Z    double  % latent codes  
            end
            
            % create a fine-grained time span from the existing basis
            tFine = linspace( obj.tSpan(1), obj.tSpan(end), ...
                              (length(obj.tSpan)-1)*10+1 );
        
            % create the set of points from the mean for each curve
            nRows = size( Z, 1 );
            XPts = repmat( eval_fd( tFine, obj.meanFd ), 1, nRows );
        
            % linearly combine the components, points-wise
            HPts = eval_fd( tFine, obj.compFd );
            for k = 1:obj.nChannels
                for j = 1:obj.nFeatures        
                    for i = 1:nRows
                        XPts(:,i,k) = XPts(:,i,k) + Z(i,j,k)*HPts(:,j,k);
                    end
                end
            end
        
            % create the functional data object
            % (ought to be a better way than providing additional parameters)
            XFdPar = fdPar( obj.basisFd, obj.penaltyOrder, obj.lambda );
            XHatFd = smooth_basis( tFine, XPts, XFdPar );

        end


    end

end



