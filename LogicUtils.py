import pandas



def get_transformed_att_value (data, att_names, gate):
    first_value = True
    for att in att_names:
        if not isinstance(att, str):
            att = att.to_string()
        if first_value:
            att_val_column = "gate.f(" + data[att].replace(False, "False").replace(True, "True")
            first_value = False
        else:
            att_val_column += "," + data[att].replace(False, "False").replace(True, "True")
    att_val_column += ")"

    transformed_column = []
    for att_val in att_val_column:
        tmp = eval(att_val)
        if tmp:
            transformed_column.append(True)
        else:
            transformed_column.append(False)
    return transformed_column

def get_input_names(data):
    return data.columns[pandas.Series(data.columns).str.startswith('i')]

def get_output_names(data):
    return data.columns[pandas.Series(data.columns).str.startswith('o')]