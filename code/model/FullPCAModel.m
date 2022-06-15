classdef FullPCAModel < FullRepresentationModel
    % Class defining a PCA model

    properties
        PCAFdParams         % functional data parameters for PCA
        PCATSpan            % time span specifically for PCA
    end

    methods

        function self = FullPCAModel( thisDataset, superArgs, args )
            % Initialize the model
            arguments
                thisDataset     ModelDataset
                superArgs.?FullRepresentationModel
                args.name           string
                args.path           string
            end

            superArgsCell = namedargs2cell(superArgs);
            argsCell = namedargs2cell(args);
            self@FullRepresentationModel( thisDataset, ...
                                          superArgsCell{:}, ...
                                          argsCell{:}, ...
                                          NumCompLines = 2 );

            self.PCATSpan = thisDataset.TSpan.Regular;
            self.PCAFdParams = thisDataset.FDA.FdParamsRegular;

        end


        function thisModel = initSubModel( self, id )
            % Initialize a sub-model
            arguments
                self            FullPCAModel
                id              double
            end

            thisModel = CompactPCAModel( self, id );

        end


    end

end



