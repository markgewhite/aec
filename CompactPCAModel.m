classdef CompactPCAModel < CompactRepresentationModel
    % Class defining an individual PCA model

    properties
        FdParams              % functional data parameters
        TSpan         double  % time span
        MeanFd                % mean functional curve
        CompFd                % component functional curves
        VarProp       double  % explained variance
        ZStd                  % latent score standard deviation (scaling factor)
        AuxModel              % auxiliary model
    end

    methods

        function self = CompactPCAModel( theFullModel )
            % Initialize the model
            arguments
                theFullModel        FullPCAModel
            end

            self@CompactRepresentationModel( theFullModel );
         
            self.TSpan = theFullModel.TSpan;
            self.FdParams = theFullModel.FdParams;

            self.MeanFd = [];
            self.CompFd = [];
            self.VarProp = [];
            self.AuxModel = [];

        end


        function self = train( self, thisTrnData, thisValData )
            % Run FPCA for the encoder
            arguments
                self            CompactPCAModel
                thisTrnData     modelDataset
                thisValData     modelDataset % redundant for PCA
            end

            % create a functional data object with fewer bases
            XFd = smooth_basis( self.TSpan, ...
                                thisTrnData.XInputRegular, ...
                                self.FdParams );

            pcaStruct = pca_fd( XFd, self.ZDim );

            self.MeanFd = pcaStruct.meanfd;
            self.CompFd = pcaStruct.harmfd;
            self.VarProp = pcaStruct.varprop;

            if size( pcaStruct.harmscr, 3 ) == 1
                pcaStruct.harmscr = permute( pcaStruct.harmscr, [1 3 2] );
            end
            self.ZStd = squeeze( std(pcaStruct.harmscr) );

            % train the auxiliary model
            Z = reshape( pcaStruct.harmscr, size(pcaStruct.harmscr, 1), [] );
            switch self.AuxModelType
                case 'Logistic'
                    self.AuxModel = fitclinear( Z, thisTrnData.Y, ...
                                                Learner = "logistic");
                case 'Fisher'
                    self.AuxModel = fitcdiscr( Z, thisTrnData.Y );
                case 'SVM'
                    self.AuxModel = fitcecoc( Z, thisTrnData.Y );
            end

        end


        function [ XC, offsets ] = latentComponents( self, Z, args )
            % Present the FPCs in form consistent with autoencoder model
            arguments
                self            CompactPCAModel
                Z               double % redundant
                args.sampling   char ...
                    {mustBeMember(args.sampling, ...
                        {'Random', 'Fixed'} )} = 'Random'
                args.nSample    double {mustBeInteger} = 0
                args.centre     logical = true
                args.range      double {mustBePositive} = 2.0
                args.convert    logical = false % redundant
            end

            if args.nSample > 0
                nSample = args.nSample;
            else
                nSample = self.NumCompLines;
            end
           
            if args.centre
                XCMean = zeros( self.TSpan, 1 );
            else
                XCMean = eval_fd( self.TSpan, self.MeanFd );
            end

            XCStd = zeros( length(self.TSpan), self.ZDim, self.XChannels );
            % compute the components
            for i = 1:self.ZDim
                XC = squeeze(eval_fd( self.TSpan, self.CompFd(i) ));
                for c = 1:self.XChannels
                    XCStd(:,i,c) = self.ZStd(i,c)*XC(:,c);
                end
            end

            if strcmp( args.sampling, 'Fixed' )
                % define the offset spacing
                offsets = linspace( -args.range, args.range, nSample );
            end

            if args.centre
                XC = zeros( length(self.TSpan), self.XChannels, self.ZDim*nSample );
            else
                XC = zeros( length(self.TSpan), self.XChannels, self.ZDim*nSample+1 );
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


        function loss = getReconLoss( self, X, XHat )
            % Calculate the reconstruction loss
            arguments
                self        CompactPCAModel
                X           double
                XHat        double
            end

            if size( X, 3 ) == 1
                loss = mean( (X-XHat).^2, [1 2] )/self.Scale;
            else
                loss = mean(mean( (X-XHat).^2, [1 2] )./ ...
                                    permute(self.Scale, [1 3 2]));
            end
    
        end


        function loss = getReconTemporalLoss( self, X, XHat )
            % Calculate the reconstruction loss across time domain
            arguments
                self        CompactPCAModel
                X           double
                XHat        double
            end

            if size( X, 3 ) == 1
                loss = squeeze(mean( (X-XHat).^2, 2 )/self.Scale);
            else
                loss = squeeze(mean( (X-XHat).^2, 2 )./ ...
                                    permute(self.Scale, [1 3 2]));
            end
    
        end


        function Z = encode( self, data )
            % Encode features Z from X using the model
            arguments
                self        CompactPCAModel
                data               
            end

            if isa( data, 'fd' )
                % validity of the FD object
                if ~isequal( self.FdParams, thisDataset.fda.XInputRegular )
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
                XFd = smooth_basis( self.TSpan, X, self.FdParams );

            end

            Z = pca_fd_score( XFd, self.MeanFd, self.CompFd, ...
                              self.ZDim, true );

            
        end

        function XHat = reconstruct( self, Z, arg )
            % Reconstruct X from Z using the model
            arguments
                self                CompactPCAModel
                Z                   double  % latent codes
                arg.convert         logical % redundant
            end
            
            % create the set of points from the mean for each curve
            nRows = size( Z, 1 );
            XHat = repmat( eval_fd( self.TSpan, self.MeanFd ), 1, nRows );
        
            % linearly combine the components, points-wise
            XC = eval_fd( self.TSpan, self.CompFd );
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



