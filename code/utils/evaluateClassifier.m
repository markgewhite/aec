function eval = evaluateClassifier( Y, YHat )
    % Evaluate predictions from a classifier
    arguments
        Y           double
        YHat        double
    end

    % generate the confusion matrix
    C = confusionmat( Y, YHat );

    % calculate the metrics from the confusion matrix (macro scores)
    eval.Accuracy = trace(C)/sum(C,'all');
    eval.ErrorRate = 1 - eval.Accuracy;

    if length(unique(Y))>2
        eval.PrecisionMicro = diag(C)./sum(C,2);
        eval.PrecisionMicro( isnan(eval.PrecisionMicro) ) = 0;
        eval.RecallMicro = diag(C)./sum(C,1)';
        eval.RecallMicro( isnan(eval.RecallMicro) ) = 0;
        eval.F1ScoreMicro = 2*eval.PrecisionMicro.*eval.RecallMicro./ ...
                                (eval.PrecisionMicro+eval.RecallMicro);
        eval.F1ScoreMicro( isnan(eval.F1ScoreMicro) ) = 0;
        eval.Precision = mean( eval.PrecisionMicro );
        eval.Recall = mean( eval.RecallMicro );
        eval.F1Score = mean( eval.F1ScoreMicro );
    else
        tp = C(2,2);
        fp = C(1,2);
        fn = C(2,1);
        eval.Precision = tp/(tp+fp);
        eval.Recall = tp/(tp+fn);
        eval.F1Score = 2*eval.Precision*eval.Recall/ ...
                                (eval.Precision+eval.Recall);
    end

end