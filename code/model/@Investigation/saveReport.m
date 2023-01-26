function report = saveReport( self )
    % Save a summary report of the investigation
    arguments
        self        Investigation
    end

    report = self.getResults;
    
    name = strcat( self.Name, "-InvestigationReport" );
    save( fullfile( self.Path, name ), 'report' );

end