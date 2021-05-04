def remove_punct(ls_with_puncts):
    puncts = ['.','!','?',',',';',':','[', ']', '{', '}', '(', ')', '\'', '\"']
    puncts_set = set(puncts)
    toRet = []
    for token in ls_with_puncts:
        if token not in puncts_set:
            toRet.append(token)

    return toRet

def has_converged(curr_record, prev_record, threshold):
    for key, val in curr_record.items():
        if abs(val - prev_record[key]) > threshold:
            return False
    return True
