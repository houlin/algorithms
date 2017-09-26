import math
import operator


def create_data_set():
    data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def calc_shannon_ent(data_set):
    set_size = len(data_set)
    labels_count = {}

    for d in data_set:
        if d[-1] not in labels_count:
            labels_count[d[-1]] = 1
        else:
            labels_count[d[-1]] += 1
    sn_entropy = 0.0
    for label in labels_count:
        p = labels_count[label] / set_size
        sn_entropy -= p * math.log(p, 2)

    return sn_entropy


def split_data_set(data_set, f, value):
    re_data_set = []
    for vec in data_set:
        if vec[f] == value:
            reduced_vec = vec[:f]
            reduced_vec.extend(vec[f + 1:])
            re_data_set.append(reduced_vec)
    return re_data_set


def choose_bast_feature_to_split(data_set):
    feature_num = len(data_set[0]) - 1
    data_set_size = len(data_set)

    f_entropy_dict = {}
    for f in range(feature_num):
        f_v_set = set()
        for vec in data_set:
            f_v_set.add(vec[f])

        f_entropy_dict[f] = 0.0
        for f_v in f_v_set:
            sub_data_set = split_data_set(data_set, f, f_v)
            p = len(sub_data_set) / float(data_set_size)
            f_entropy_dict[f] += p * calc_shannon_ent(sub_data_set)

    min_f_entropy = 0.0
    min_f = 0
    for _f in f_entropy_dict:
        if f_entropy_dict[_f] < min_f_entropy:
            min_f_entropy = f_entropy_dict[_f]
            min_f = _f
    return min_f


# data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
# labels = ['no surfacing', 'flippers']


def dec_tree(r):
    return {'v': r}


def insert_child(f_tree, f_v, sub_tree):
    f_tree[f_v] = sub_tree


def get_most_class(data_set):
    most_class_count = {}
    for c in [vec[-1] for vec in data_set]:
        if c not in most_class_count:
            most_class_count[c] = 0
        else:
            most_class_count[c] += 1
    return max(most_class_count, key=lambda k: most_class_count[k])


def get_feature_value_set(data_set, f):
    f_v_list = [vec[f] for vec in data_set]
    return set(f_v_list)


def create_tree(data_set):
    ent = calc_shannon_ent(data_set)
    if ent <= 0.5:
        return dec_tree(get_most_class(data_set))

    best_f = choose_bast_feature_to_split(data_set)
    best_f_v_set = get_feature_value_set(data_set, best_f)

    f_tree = dec_tree(best_f)
    for f_v in best_f_v_set:
        sub_data_set = split_data_set(data_set, best_f, f_v)
        sub_tree = create_tree(sub_data_set)
        insert_child(f_tree, f_v, sub_tree)

    return f_tree


def classify(my_dec_tree, f_vec):
    f = my_dec_tree['v']
    v = f_vec[f]
    if len(my_dec_tree[v]) > 1:
        return classify(my_dec_tree[v], f_vec)
    else:
        return my_dec_tree[v]['v']


if __name__ == "__main__":
    my_dat, labels = create_data_set()
    # print(my_dat)
    my_dec_tree = create_tree(my_dat)
    # print(my_dec_tree)
    print(classify(my_dec_tree, [1, 0]))
