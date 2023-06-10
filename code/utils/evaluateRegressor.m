function eval = evaluateRegressor( Y, YHat )
    % Evaluate predictions from a regressor
    arguments
        Y           double
        YHat        double
    end

    % calculate the metrics
    eval.RMSE = sqrt(mean( (YHat - Y).^2 ));
    eval.MAE = mean( abs(YHat - Y) );

end