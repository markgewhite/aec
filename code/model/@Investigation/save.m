function report = save( self )
    % Save the investigation to a specified path
    arguments
        self        Investigation
    end

    report = self.getResults;
    
    name = strcat( self.Name, "-Investigation" );
    save( fullfile( self.Path, name ), 'report' );

end