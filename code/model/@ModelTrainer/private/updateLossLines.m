function updateLossLines( lossLines, j, newPts )
    % Update loss animated lines
    arguments
        lossLines
        j               double
        newPts          double
    end

    for i = 1:length(lossLines)
        addpoints( lossLines(i), j, newPts(i) );
    end
    drawnow;

end