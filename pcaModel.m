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
        penaltyOrder          % linear differential operator
    end

    methods

        function self = pcaModel( fdParameters, superArgs )
            % Initialize the model
            arguments
                fdParameters  % functional data parameters selfect
                superArgs.?representationModel
            end

            argsCell = namedargs2cell(superArgs);
            self = self@representationModel( argsCell{:} );

            self.meanFd = [];
            self.compFd = [];
            self.varProp = [];

            self.basisFd = getbasis( fdParameters );
            self.tSpan = getbasispar( self.basisFd );
            self.nKnots = getnbasis( self.basisFd );
            self.lambda = getlambda( fdParameters );
            self.penaltyOrder = getLfd( fdParameters );

        end


        function self = train( self, XFd )
            % Run FPCA for the encoder
            arguments
                self
                XFd {mustBeValidBasis(self, XFd)}  
            end

            pcaStruct = pca_fd( XFd, self.ZDim );

            self.meanFd = pcaStruct.meanfd;
            self.compFd = pcaStruct.harmfd;
            self.varProp = pcaStruct.varprop;

        end

        function Z = encode( self, XFd )
            % Encode features Z from X using the model
            arguments
                self
                XFd {mustBeValidBasis(self, XFd)}  
            end

            Z = pca_fd_score( XFd, self.meanFd, self.compFd, ...
                              self.ZDim, true );

        end

        function XHatFd = reconstruct( self, Z )
            % Reconstruct X from Z using the model
            arguments
                self
                Z    double  % latent codes  
            end
            
            % create a fine-grained time span from the existing basis
            tFine = linspace( self.tSpan(1), self.tSpan(end), ...
                              (length(self.tSpan)-1)*10+1 );
        
            % create the set of points from the mean for each curve
            nRows = size( Z, 1 );
            XPts = repmat( eval_fd( tFine, self.meanFd ), 1, nRows );
        
            % linearly combine the components, points-wise
            HPts = eval_fd( tFine, self.compFd );
            for k = 1:self.XChannels
                for j = 1:self.ZDim        
                    for i = 1:nRows
                        XPts(:,i,k) = XPts(:,i,k) + Z(i,j,k)*HPts(:,j,k);
                    end
                end
            end
        
            % create the functional data selfect
            % (ought to be a better way than providing additional parameters)
            XFdPar = fdPar( self.basisFd, self.penaltyOrder, self.lambda );
            XHatFd = smooth_basis( tFine, XPts, XFdPar );

        end


    end

end



