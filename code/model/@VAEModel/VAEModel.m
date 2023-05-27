classdef VAEModel < AEModel
    % Subclass implementing a variational autoencoder

    properties
        NumEncodingDraws     % number of draws from the distribution per row
        UseEncodingMean      % flag whether to use mean output in predictions
    end

    methods

        function self = VAEModel( thisDataset, ...
                                  superArgs, ...
                                  args )

            % Initialize the variational autoencoder
            arguments
                thisDataset              ModelDataset
                superArgs.name           string
                superArgs.path           string
                superArgs.?AEModel
                args.NumEncodingDraws    double ...
                        {mustBeInteger,mustBePositive} = 1
                args.UseEncodingMean     logical = true
            end

            superArgsCell = namedargs2cell( superArgs );
           
            self = self@AEModel( thisDataset, superArgsCell{:} );

            self.NumEncodingDraws = args.NumEncodingDraws;
            self.UseEncodingMean = args.UseEncodingMean;

        end

    end

end

