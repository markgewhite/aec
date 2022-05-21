classdef pcaModel < representationModel
    % Class defining a PCA model

    properties
        meanFd                % mean curve
        compFd                % functional principal components
        ZStd                  % latent score standard deviation (scaling factor)
        varProp       double  % explained variance
        fdParams              % functional data parameters
        tSpan         double  % time span

        auxModelType          % type of auxiliary model to use
        auxModel              % auxiliary model itself
    end

    methods

        function self = pcaModel( XChannels, ZDim, superArgs, args )
            % Initialize the model
            arguments
                XChannels       double {mustBeInteger, mustBePositive}
                ZDim            double {mustBeInteger, mustBePositive}
                superArgs.?representationModel
                args.auxModel       string ...
                        {mustBeMember( args.auxModel, ...
                                {'Fisher', 'SVM'} )} = 'Fisher'
            end

            argsCell = namedargs2cell(superArgs);
            self = self@representationModel( argsCell{:}, ...
                                             ZDim = ZDim, ...
                                             XChannels = XChannels, ...
                                             NumCompLines = 2 );


            self.meanFd = [];
            self.compFd = [];
            self.varProp = [];

            self.auxModelType = args.auxModel;

        end


        function self = train( self, thisDataset )
            % Run FPCA for the encoder
            arguments
                self            pcaModel
                thisDataset     modelDataset
            end

            self.fdParams = thisDataset.fda.fdParamsRegular;
            self.tSpan = thisDataset.tSpan.regular;

            XInput = thisDataset.XInputRegular;
            %XInput = permute( XInput, [1 3 2] );

            % convert input to a functional data object
            XFd = smooth_basis( self.tSpan, XInput, self.fdParams );

            pcaStruct = pca_fd( XFd, self.ZDim );

            self.meanFd = pcaStruct.meanfd;
            self.compFd = pcaStruct.harmfd;
            self.varProp = pcaStruct.varprop;

            if size( pcaStruct.harmscr, 3 ) == 1
                pcaStruct.harmscr = permute( pcaStruct.harmscr, [1 3 2] );
            end
            self.ZStd = squeeze( std(pcaStruct.harmscr) );

            % train the auxiliary model
            Z = reshape( pcaStruct.harmscr, size(pcaStruct.harmscr, 1), [] );
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
                Z               double % redundant
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

            XCStd = zeros( length(self.tSpan), self.ZDim, self.XChannels );
            % compute the components
            for i = 1:self.ZDim
                XC = squeeze(eval_fd( self.tSpan, self.compFd(i) ));
                for c = 1:self.XChannels
                    XCStd(:,i,c) = self.ZStd(i,c)*XC(:,c);
                end
            end

            if strcmp( args.sampling, 'Fixed' )
                % define the offset spacing
                offsets = linspace( -args.range, args.range, nSample );
            end

            if args.centre
                XC = zeros( length(self.tSpan), self.XChannels, self.ZDim*nSample );
            else
                XC = zeros( length(self.tSpan), self.XChannels, self.ZDim*nSample+1 );
                XC(:,:,end) = XCMean;
            end
            
            for j = 1:nSample
                
                switch args.sampling
                    case 'Random'
                        offset = args.range*rand;
                    case 'Fixed'
                        offset = offsets(j);
                end

                for i =1:self.ZDim
                    XC(:,:,(i-1)*nSample+j) = XCMean + offset*XCStd(:,i,:);
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
                if ~isequal( self.fdParams, thisDataset.fda.XInputRegular )
                    eid = 'PCAModel:InvalidFDParam';
                    msg = 'The input FD parameters do not match the model''s FD parameters.';
                    throwAsCaller( MException(eid,msg) );
                end
                XFd = data;

            else
                if isa( data, 'modelDataset' )
                    X = data.XInputRegular;
                    %X = permute( X, [1 3 2] );

                elseif isa( data, 'double' )
                    X = data;

                else
                    eid = 'PCAModel:InvalidData';
                    msg = 'The input data is not a class modelDataset or double.';
                    throwAsCaller( MException(eid,msg) );

                end
                % convert input to a functional data object
                XFd = smooth_basis( self.tSpan, X, self.fdParams );

            end

            Z = pca_fd_score( XFd, self.meanFd, self.compFd, ...
                              self.ZDim, true );

            
        end

        function XHat = reconstruct( self, Z, arg )
            % Reconstruct X from Z using the model
            arguments
                self                pcaModel
                Z                   double  % latent codes
                arg.convert         logical = true
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

            %if arg.convert
            %    XHat = permute( XHat, [1 3 2] );
            %end
 
        end


    end

end



