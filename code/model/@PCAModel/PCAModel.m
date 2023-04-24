classdef PCAModel < RepresentationModel
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

        function self = PCAModel( thisDataset, superArgs, args )
            % Initialize the model
            arguments
                thisDataset         ModelDataset
                superArgs.?RepresentationModel
                args.name           string
                args.path           string
            end

            superArgs.ComponentType = 'PDP';
            superArgsCell = namedargs2cell(superArgs);
            argsCell = namedargs2cell(args);
            self@RepresentationModel( thisDataset, ...
                                      superArgsCell{:}, ...
                                      argsCell{:} );

            self.PCATSpan = thisDataset.TSpan.Regular;
            self.PCAFdParams = thisDataset.FDA.FdParamsRegular;

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
