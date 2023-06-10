function reportResult( eval, cor )

    disp(['         Reconstruction Loss = ' ...
        num2str( eval.ReconLoss, '%.3f' )]);
    disp(['Smoothed Reconstruction Loss = ' ...
        num2str( eval.ReconLossSmoothed, '%.3f' )]);

    if isfield( eval, 'AuxModelErrorRate' )
        disp(['  Auxiliary Model Error Rate = ' ...
            num2str( eval.AuxModelErrorRate, '%.3f' )]);
        disp(['    Auxiliary Model F1 Score = ' ...
            num2str( eval.AuxModelF1Score, '%.3f' )]);
    end

    if isfield( eval, 'AuxNetworkErrorRate' )
        disp(['Auxiliary Network Error Rate = ' ...
            num2str( eval.AuxNetworkErrorRate, '%.3f' )]);
        disp(['  Auxiliary Network F1 Score = ' ...
            num2str( eval.AuxNetworkF1Score, '%.3f' )]);
    end

    if isfield( eval, 'AuxModelRMSE' )
        disp(['        Auxiliary Model RMSE = ' ...
            num2str( eval.AuxModelRMSE, '%.3f' )]);
    end

    if isfield( eval, 'AuxNetworkRMSE' )
        disp(['      Auxiliary Network RMSE = ' ...
            num2str( eval.AuxNetworkRMSE, '%.3f' )]);
    end

    if isfield( eval, 'ComparatorErrorRate' )
        disp(['       Comparator Error Rate = ' ...
            num2str( eval.ComparatorErrorRate, '%.3f' )]);
        disp(['         Comparator F1 Score = ' ...
            num2str( eval.ComparatorF1Score, '%.3f' )]);
    end

    disp(['               Z Correlation = ' ...
        num2str( cor.ZCorrelation, '%.3f' )]);
    disp(['              XC Correlation = ' ...
        num2str( cor.XCCorrelation, '%.3f' )]);

end