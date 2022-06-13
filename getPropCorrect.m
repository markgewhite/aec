function err = getPropCorrect( Y, YHat )
    % Compute the proportion correct
    arguments
        Y           double
        YHat        double
    end

    err = 1 - mean( YHat==Y );

end