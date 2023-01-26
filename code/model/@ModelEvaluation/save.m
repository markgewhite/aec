function save( thisEvaluation )
    % Save the evaluation
    arguments
        thisEvaluation        ModelEvaluation
    end
   
    filename = strcat( thisEvaluation.Name, "-Evaluation" );
    save( fullfile( thisEvaluation.Path, filename ), 'thisEvaluation' );

end   
