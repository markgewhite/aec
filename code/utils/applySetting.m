function setup = applySetting( setup, parameter, value )
    % Apply the parameter value by recursively moving through structure
    arguments
        setup       struct
        parameter   string
        value       
    end

    var = extractBefore( parameter, "." );
    remainder = extractAfter( parameter, "." );
    if contains( remainder, "." )
        setup.(var) = applySetting( setup.(var), remainder, value );
    else
        switch class( value )
            case {'double', 'char', 'string', 'logical'}
                setup.(var).(remainder) = value;
            case 'cell'
                setup.(var).(remainder) = value{1};
            case {'categorical'}
                if value=='false' %#ok<BDSCA> 
                    setup.(var).(remainder) = false;
                elseif value=='true' %#ok<BDSCA> 
                    setup.(var).(remainder) = true;
                else
                    setup.(var).(remainder) = value;
                end
            otherwise
                msg = strcat( "Value type for setup.", ...
                        var, ".", remainder, " not recognised.");
                error( msg );
        end
    end

end