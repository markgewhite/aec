classdef ParallelInvestigation < Investigation
    % Class defining an investigation grid search run in parallel

    properties
    end

    methods

        function self = ParallelInvestigation( name, path, parameters, ...
                                       searchValues, setup, ...
                                       catchErrors, memorySaving )
            % Construct an investigation comprised of evaluations
            arguments
                name            string
                path            string
                parameters      string
                searchValues
                setup           struct
                catchErrors     logical = false
                memorySaving    double {mustBeInteger, ...
                                mustBeInRange( memorySaving, 0, 3 )} = 0
            end

            setup.model.args.ShowPlots = false;
            setup.eval.args.Verbose = false;

            self@Investigation( name, path, parameters, ...
                                searchValues, setup, ...
                                catchErrors, memorySaving );

        end

    end

end