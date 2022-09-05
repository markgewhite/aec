function XLen = adjustXLengths( XLen, tSpan, tSpanAdaptive, padding )

    for i = 1:length(XLen)
        switch padding

            case 'Left'
                tEnd = tSpan( length(tSpan)-XLen(i)+1 );
                XLen(i) = length(tSpanAdaptive) - find( tEnd <= tSpanAdaptive, 1 ) + 1;

            case {'Right', 'Both'}
                tEnd = tSpan( XLen(i) );
                XLen(i) = find( tEnd <= tSpanAdaptive, 1 );

        end
    end

end