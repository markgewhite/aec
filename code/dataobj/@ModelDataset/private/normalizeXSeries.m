function XN = normalizeXSeries( X, nPts, type, pad )

    switch type
    
        case 'LTN' % time normalization
            XN = timeNormalize( X, nPts );

        case 'PAD' % padding
            XN = padData( X, pad.Length, pad.Value, ...
                             Same = pad.Same, ...
                             Location = pad.Location, ...
                             Anchoring = pad.Anchoring );
            XN = timeNormalize( XN, nPts );
    
    end

end