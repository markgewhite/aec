classdef SubPCAModel < SubRepresentationModel
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

        function self = SubPCAModel( theFullModel, fold )
            % Initialize the model
            arguments
                theFullModel        FullPCAModel
                fold                double
            end

            self@SubRepresentationModel( theFullModel, fold );
         
            self.PCATSpan = theFullModel.PCATSpan;
            self.PCAFdParams = theFullModel.PCAFdParams;

            self.MeanFd = [];
            self.CompFd = [];
            self.VarProp = [];
            self.AuxModel = [];

        end

        % class methods

        [ XC, XMean, offsets ] = calcLatentComponents( self, Z, args )

        Z = encode( self, data, args )

        self = train( self, thisData )

        [ XHat, XHatSmth, XHatReg ] = reconstruct( self, Z, args )
    
    end

end
