classdef ReloadedInvestigation < Investigation
    % Class defining an investigation reloaded from a saved report

    properties
    end

    methods

        function self = ReloadedInvestigation( report, args )
            % Construct an investigation shell
            arguments
                report          struct
                args.Name       string
                args.Path       string
            end

            if isfield( report, 'Name' )
                name = report.name;
            elseif isfield( args, 'Name' )
                name = args.Name;
            else
                name = 'Reloaded';
            end
            
            if isfield( report, 'Path' )
                path = report.path;
            elseif isfield( args, 'Path' )
                path = args.Path;
            else
                path = report.BaselineSetup.model.args.path;
            end

            setup = report.BaselineSetup;
            parameters = report.Parameters;
            searchValues = report.GridSearch;
            catchErrors = true;
            memorySaving = 1;

            self@Investigation( name, path, parameters, ...
                                searchValues, setup, ...
                                catchErrors, memorySaving );

            if isfield( report, 'IsComplete' )
                self.IsComplete = report.IsComplete;
            else
                self.IsComplete = true( self.NumEvaluations, 1 );
            end

            self.TrainingResults = report.TrainingResults;
            self.TestingResults = report.TestingResults;


        end

    end

end