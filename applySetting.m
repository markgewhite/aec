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
            case {'double', 'string', 'logical'}
                setup.(var).(remainder) = value;
            case 'cell'
                setup.(var).(remainder) = value{1};
        end
    end

end