classdef pcaModel < representationModel
    % Class defining a PCA model

    properties
        meanFd                % mean curve
        compFd                % functional principal components
        varProp       double  % explained variance
        fdParams              % functional data parameters
        basisFd               % functional basis
        tSpan         double  % time span
        nKnots        double  % number of knots
        lambda        double  % roughness penalty
        penaltyOrder          % linear differential operator

        auxModelType          % type of auxiliary model to use
        auxModel              % auxiliary model itself
    end

    methods

        function self = pcaModel( fdParameters, superArgs, args )
            % Initialize the model
            arguments
                fdParameters
                superArgs.?representationModel
                args.auxModel       string ...
                        {mustBeMember( args.auxModel, ...
                                {'Fisher', 'SVM'} )} = 'Fisher'
            end

            argsCell = namedargs2cell(superArgs);
            self = self@representationModel( argsCell{:}, ...
                                             NumCompLines = 2 );


            self.meanFd = [];
            self.compFd = [];
            self.varProp = [];

            self.fdParams = fdParameters;

            self.basisFd = getbasis( fdParameters );
            range = getbasisrange( self.basisFd );
            knots = getbasispar( self.basisFd );
            self.tSpan = [ range(1) knots range(2) ];
            self.nKnots = getnbasis( self.basisFd );
            self.lambda = getlambda( fdParameters );
            self.penaltyOrder = getLfd( fdParameters );

            self.auxModelType = args.auxModel;

        end


        function self = train( self, thisDataset )
            % Run FPCA for the encoder
            arguments
                self            pcaModel
                thisDataset     modelDataset
            end

            if ~isequal( self.fdParams, thisDataset.fda.fdParams )
                eid = 'PCAModel:InvalidFDParam';
                msg = 'The dataset''s FD parameters do not match those of this initialized model.';
                throwAsCaller( MException(eid,msg) );
            end

            % convert input to a functional data object
            XFd = smooth_basis( thisDataset.fda.tSpan, ...
                                thisDataset.XTarget, ...
                                thisDataset.fda.fdParams );

            pcaStruct = pca_fd( XFd, self.ZDim );

            self.meanFd = pcaStruct.meanfd;
            self.compFd = pcaStruct.harmfd;
            self.varProp = pcaStruct.varprop;

            % train the auxiliary model
            Z = pcaStruct.harmscr;
            switch self.auxModelType
                case 'Fisher'
                    self.auxModel = fitcdiscr( Z, thisDataset.Y );
                case 'SVM'
                    self.auxModel = fitcecoc( Z, thisDataset.Y );
            end

        end


        function XC = latentComponents( self, Z, args )
            % Present the FPCs in form consistent with autoencoder model
            arguments
                self            pcaModel
                Z               double
                args.sampling   char ...
                    {mustBeMember(args.sampling, ...
                        {'Random', 'Fixed'} )} = 'Random'
                args.nSample    double {mustBeInteger} = 0
                args.centre     logical = true
                args.range      double {mustBePositive} = 2.0
            end

            if args.nSample > 0
                nSample = args.nSample;
            else
                nSample = self.NumCompLines;
            end
           
            if args.centre
                XCMean = zeros( self.tSpan, 1 );
            else
                XCMean = eval_fd( self.tSpan, self.meanFd );
            end

            XCStd = zeros( length(self.tSpan), self.ZDim );
            % compute the components
            for i = 1:self.ZDim
               XCStd(:,i) = eval_fd( self.tSpan, self.compFd(i) );
            end

            if strcmp( args.sampling, 'Fixed' )
                % define the offset spacing
                offsets = linspace( -args.range, args.range, nSample );
            end

            if args.centre
                XC = zeros( length(self.tSpan), self.ZDim*nSample );
            else
                XC = zeros( length(self.tSpan), self.ZDim*nSample+1 );
                XC(:,end) = XCMean;
            end
            
            for j = 1:nSample
                
                switch args.sampling
                    case 'Random'
                        offset = args.range*rand;
                    case 'Fixed'
                        offset = offsets(j);
                end

                for i =1:self.ZDim
                    XC(:,(i-1)*nSample+j) = XCMean + offset*XCStd(:,i);
                end

            end

        end


    end

    
    methods (Static)

        function Z = encode( self, data )
            % Encode features Z from X using the model
            arguments
                self        pcaModel
                data               
            end

            if isa( data, 'fd' )
                % validity of the FD object
                if ~isequal( self.fdParams, thisDataset.fda.fdParams )
                    eid = 'PCAModel:InvalidFDParam';
                    msg = 'The input FD parameters do not match the model''s FD parameters.';
                    throwAsCaller( MException(eid,msg) );
                end
                XFd = data;

            else
                if isa( data, 'modelDataset' )
                    X = data.XTarget;

                elseif isa( data, 'double' )
                    X = data;

                else
                    eid = 'PCAModel:InvalidData';
                    msg = 'The input data is not a class modelDataset or double.';
                    throwAsCaller( MException(eid,msg) );

                end
                % convert input to a functional data object
                XFd = smooth_basis( self.tSpan, ...
                                    X, ...
                                    self.fdParams );

            end

            Z = pca_fd_score( XFd, self.meanFd, self.compFd, ...
                              self.ZDim, true );

            
        end

        function XHat = reconstruct( self, Z )
            % Reconstruct X from Z using the model
            arguments
                self        pcaModel
                Z           double  % latent codes  
            end
            
            % create the set of points from the mean for each curve
            nRows = size( Z, 1 );
            XHat = repmat( eval_fd( self.tSpan, self.meanFd ), 1, nRows );
        
            % linearly combine the components, points-wise
            XC = eval_fd( self.tSpan, self.compFd );
            for k = 1:self.XChannels
                for j = 1:self.ZDim        
                    for i = 1:nRows
                        XHat(:,i,k) = XHat(:,i,k) + Z(i,j,k)*XC(:,j,k);
                    end
                end
            end
 
        end


    end

end



