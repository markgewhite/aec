function updateLossLines( lossLines, data )
    % Update loss animated lines
    arguments
        lossLines
        data            double
    end

    j = data(1);
    newPts = data(2:end);

    for i = 1:length(lossLines)
        addpoints( lossLines(i), j, newPts(i) );
    end
    drawnow;

end