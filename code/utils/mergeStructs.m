function z = mergeStructs( x, y )
    % Merge structures from two variables - output has all fields

    z = cell2struct( [struct2cell(x); struct2cell(y)], ...
                     [fieldnames(x); fieldnames(y)] );

end