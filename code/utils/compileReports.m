function investigations = compileReports( names )
    % Compile a cell array of ReloadedInvestigation from reports
    arguments
        names           string
    end

    % set the source path
    path = fileparts( which('code/utils/compileReports.m') );
    path = [path '/../../results/grid/'];

    % load the reports and create the investigations
    numReports = length( names );
    investigations = cell( numReports, 1 );
    for i = 1:numReports
        filename = strcat( names(i), '-InvestigationReport.mat' );
        load( fullfile(path, filename), 'report');
        investigations{i} = ReloadedInvestigation( report, name = names(i) );
    end
   

end