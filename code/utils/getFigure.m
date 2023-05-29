function fig = getFigure( n )
    % Find an existing figure with the required number or create it

    allFigs = findall( groot, Type = 'Figure');
    
    fig = [];
    
    for i = 1:length(allFigs)
        if allFigs(i).Number==n
            break
        end
    end
    
    if isempty( fig )
        fig = figure(n);
    else
        fig = figure(i);
    end

end