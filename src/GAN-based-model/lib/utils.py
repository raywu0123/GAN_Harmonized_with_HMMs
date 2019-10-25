def array_to_string(a, strip_pad=True):
    s = ' '.join(map(str, a))
    if strip_pad:
        s = s.strip('0 ')
    return s
