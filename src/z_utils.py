def sliding_windows(seq, n):
    """
    Generate sliding windows over a sequence.
    - Pads with the last element if len(seq) < n or for end windows.


    I believe there is some issue with the padding here? what's the normal approach if the window ends early? can't I just end it early?... so there are 2 sentences there, instead of 3? 


    """
    seq = list(seq)
    if not seq:
        return
    if n <= 1:
        for item in seq:
            yield [item]
    else:
        for i in range(len(seq)):
            window = seq[i:i+n]
            # Pad with last element if window too short
            while len(window) < n:
                window.append(seq[-1])
            yield window
