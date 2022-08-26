function s = flattenStruct( s )
    % Flatten a nested structure

    nested = true;
    while nested
        nested = false;
        f = fieldnames( s );
        for i = 1:length(f)
            if isstruct(s.(f{i}))
                s2 = s.(f{i});
                f2 = fieldnames( s2 );
                for j = 1:length(f2)
                    f2ext = strcat( f{i}, f2{j} );
                    if isstruct(s2.(f2{j}))
                        s.(f2ext) = flattenStruct( s2 );
                        nested = true;
                    else
                        s.(f2ext) = s2.(f2{j});
                    end
                end
                s = rmfield( s, f{i} );
            end
        end
    end

end