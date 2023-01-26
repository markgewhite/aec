function save( thisInvestigation )
    % Save the investigation object
    arguments
        thisInvestigation        Investigation
    end

    name = strcat( thisInvestigation.Name, "-Investigation" );
    save( fullfile( thisInvestigation.Path, name ), 'thisInvestigation' );

end