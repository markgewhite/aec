function name = constructName( idx )
    % Construct a name from the indices

    name = '(';
    for j = 1:length(idx)
        name = strcat( name, num2str(idx(j)) );
        if j < length(idx)
            name = strcat( name, ',' );
        end
    end
    name = strcat( name, ')');

end