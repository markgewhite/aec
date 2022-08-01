classdef CompactPCAModel < CompactRepresentationModel
    % Class defining an individual PCA model

    properties
        PCAFdParams           % functional data parameters
        PCATSpan              % time span
        MeanFd                % mean functional curve
        CompFd                % component functional curves
        VarProp               % explained variance
        ZStd                  % latent score standard deviation (scaling factor)
    end

    methods

        function self = CompactPCAModel( theFullModel, fold )
            % Initialize the model
            arguments
                theFullModel        FullPCAModel
                fold                double
            end

            self@CompactRepresentationModel( theFullModel, fold );
         
            self.PCATSpan = theFullModel.PCATSpan;
            self.PCAFdParams = theFullModel.PCAFdParams;

            self.MeanFd = [];
            self.CompFd = [];
            self.VarProp = [];
            self.AuxModel = [];

        end


        function self = train( self, thisData )
            % Run FPCA for the encoder
            arguments
                self         CompactPCAModel
                thisData     ModelDataset
            end

            % create a functional data object with fewer bases
            XFd = smooth_basis( self.TSpan.Regular, ...
                                thisData.XInputRegular, ...
                                self.FDA.FdParamsRegular );

            pcaStruct = pca_fd( XFd, self.ZDim );

            self.MeanFd = pcaStruct.meanfd;
            self.CompFd = pcaStruct.harmfd;
            self.VarProp = pcaStruct.varprop;

            if size( pcaStruct.harmscr, 3 ) == 1
                pcaStruct.harmscr = permute( pcaStruct.harmscr, [1 3 2] );
            end
            self.ZStd = squeeze( std(pcaStruct.harmscr) );

            % generate the latent components
            Z = reshape( pcaStruct.harmscr, size(pcaStruct.harmscr, 1), [] );

            % train the auxiliary model
            switch self.AuxModelType
                case 'Logistic'
                    self.AuxModel = fitclinear( Z, thisData.Y, ...
                                                Learner = "logistic");
                case 'Fisher'
                    self.AuxModel = fitcdiscr( Z, thisData.Y );
                case 'SVM'
                    self.AuxModel = fitcecoc( Z, thisData.Y );
            end

            % compute the mean curve directly
            self.MeanCurve = eval_fd( self.TSpan.Regular, ...
                                      self.MeanFd );

            % compute the components' explained variance
            [self.LatentComponents, ...
                self.VarProportion, self.ComponentVar] ...
                            = self.getLatentComponents( thisData );
            
            % plot them on specified axes
            plotLatentComp( self, type = 'Smoothed', shading = true );
        
            % plot the Z distributions
            plotZDist( self, Z );
        
            % plot the Z clusters
            plotZClusters( self, Z, Y = thisData.Y );

        end


        function [ XC, XMean, offsets ] = calcLatentComponents( self, Z, args )
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
                args.final      logical = false % redundant
            end

            % compute the components
            nSample = self.NumCompLines;
            offsets = linspace( -2, 2, nSample );
            XC = zeros( length(self.PCATSpan), self.XChannels, nSample );
            for i =1:self.ZDim
                FPC = squeeze(eval_fd( self.PCATSpan, self.CompFd(i) ));
                for c = 1:self.XChannels
                    for j = 1:nSample
                        XC(:,c,(i-1)*nSample+j) = offsets(j)*FPC(:,c);
                    end
                end
            end

            XMean = squeeze(eval_fd( self.PCATSpan, self.MeanFd ));

        end


        function Z = encode( self, data )
            % Encode features Z from X using the model
            arguments
                self        CompactPCAModel
                data               
            end

            if isa( data, 'fd' )
                % validity of the FD object
                if ~isequal( self.PCAFdParams, thisDataset.fda.XInputRegular )
                    eid = 'PCAModel:InvalidFDParam';
                    msg = 'The input FD parameters do not match the model''s FD parameters.';
                    throwAsCaller( MException(eid,msg) );
                end
                XFd = data;

            else
                if isa( data, 'ModelDataset' )
                    X = data.XInputRegular;
                    %X = permute( X, [1 3 2] );

                elseif isa( data, 'double' )
                    X = data;

                else
                    eid = 'PCAModel:InvalidData';
                    msg = 'The input data is not a class ModelDataset or double.';
                    throwAsCaller( MException(eid,msg) );

                end
                % convert input to a functional data object
                XFd = smooth_basis( self.TSpan.Regular, X, ...
                                    self.FDA.FdParamsRegular );

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
            XHat = repmat( eval_fd( self.PCATSpan, self.MeanFd ), 1, nRows );
        
            % linearly combine the components, points-wise
            XC = eval_fd( self.PCATSpan, self.CompFd );
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



