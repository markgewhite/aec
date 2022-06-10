classdef FullPCAModel < FullRepresentationModel
    % Class defining a PCA model

    properties
        FdParams              % functional data parameters
        TSpan         double  % time span
    end

    methods

        function self = FullPCAModel( ZDim, thisDataset, superArgs )
            % Initialize the model
            arguments
                ZDim            double ...
                        {mustBeInteger, mustBePositive}
                thisDataset     modelDataset
                superArgs.?FullRepresentationModel
            end

            argsCell = namedargs2cell(superArgs);
            self@FullRepresentationModel( argsCell{:}, ...
                                 ZDim = ZDim, ...
                                 XChannels = thisDataset.XChannels, ...
                                 NumCompLines = 2 );

            self.TSpan = thisDataset.TSpan.Regular;
            self.FdParams = thisDataset.FDA.FdParamsRegular;

            % set the scaling factor(s) based on all X
            self = self.setScalingFactor( thisDataset.XTarget );

            self.SubModels = cell( self.KFolds, 1 );

        end


        function thisModel = initSubModel( self )
            % Initialize a sub-model
            arguments
                self            FullPCAModel
            end

            thisModel = CompactPCAModel( self );

        end


        function self = setScalingFactor( self, data )
            % Set the scaling factors for reconstructions
            arguments
                self            FullPCAModel
                data            double
            end
            
            % set the channel-wise scaling factor
            self.Scale = squeeze(mean(var( data )))';

        end


    end

end



