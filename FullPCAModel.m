classdef FullPCAModel < FullRepresentationModel
    % Class defining a PCA model

    properties
        PCAFdParams         % functional data parameters for PCA
        PCATSpan            % time span specifically for PCA
    end

    methods

        function self = FullPCAModel( thisDataset, superArgs )
            % Initialize the model
            arguments
                thisDataset     modelDataset
                superArgs.?FullRepresentationModel
            end

            argsCell = namedargs2cell(superArgs);
            self@FullRepresentationModel( thisDataset, ...
                                          argsCell{:}, ...
                                          NumCompLines = 2 );

            self.PCATSpan = thisDataset.TSpan.Target;
            self.PCAFdParams = thisDataset.FDA.FdParamsTarget;

            self.SubModels = cell( self.KFolds, 1 );

        end


        function thisModel = initSubModel( self )
            % Initialize a sub-model
            arguments
                self            FullPCAModel
            end

            thisModel = CompactPCAModel( self );

        end


    end

end



