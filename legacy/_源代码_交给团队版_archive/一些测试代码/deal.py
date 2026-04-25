from collections import defaultdict

in_file = r'D:\大学\毕设\IS2009EmotionChallenge\IS2009EmotionChallenge\labels\word_labels_11cl_corpus.txt'
out_file = r'D:\大学\毕设\IS2009EmotionChallenge\IS2009EmotionChallenge\labels\word_labels_11cl_corpus_processed.txt'

f_in = open(in_file, 'r')
f_out = open(out_file, 'w')

labels = defaultdict(list)

for line in f_in:
    parts = line.strip().split()
    i = 0
    j = 0
    while True:
        if parts[0][i] == '_':
            j += 1
        if j == 4:
            break
        i += 1
    file_id = parts[0][:i]
    tag = parts[-5:] # Last five tags
    labels[file_id].extend(tag)

f_in.close()

filtered_labels = {}
for file_id, tag_list in labels.items():
    tag_count = defaultdict(int)
    for tag in tag_list:
        if tag != 'X':
            tag_count[tag] += 1
    # Find the tag with the maximum count
    l = len(tag_list)
    if tag_count:
        max_tag = max(tag_count, key=tag_count.get)
        if l == 5:
            if tag_count[max_tag] >= 3:
                filtered_labels[file_id] = max_tag
            else:
                filtered_labels[file_id] = 'X'  # No valid tag
        else:
            if tag_count[max_tag] >= l * (3/5) * 0.75:
                filtered_labels[file_id] = max_tag
            else:
                filtered_labels[file_id] = 'X'  # No valid tag
    else:
        filtered_labels[file_id] = 'X'  # No valid tag

label_to_files = defaultdict(list)
for filename, label in filtered_labels.items():
    label_to_files[label].append(filename)

a = []
b = [0,0,0,0,0,0,0,0,0,0,0]
flag = 1

for i in label_to_files:
    a.append(i) # a[0] = 'N'

# N: in label_to_files['N']

for i in range(707):
    for j in range(4):
        # 4 N + 2 other
        f_out.write(f"N {label_to_files[a[0]][b[0]]}\n")
        b[0] += 1
    for j in range(2):
        while b[flag] >= len(label_to_files[a[flag]]):
            flag = (flag + 1) % 11
            if flag == 0:
                flag = 1
        f_out.write(f"{a[flag]} {label_to_files[a[flag]][b[flag]]}\n")
        b[flag] += 1
        flag = (flag + 1) % 11
        if flag == 0:
            flag = 1

for i in range(2329):
    for j in range(5):
        # 5 N + 1 other
        f_out.write(f"N {label_to_files[a[0]][b[0]]}\n")
        b[0] += 1
    while b[flag] >= len(label_to_files[a[flag]]):
        flag = (flag + 1) % 11
        if flag == 0:
            flag = 1
    f_out.write(f"{a[flag]} {label_to_files[a[flag]][b[flag]]}\n")
    b[flag] += 1
    flag = (flag + 1) % 11
    if flag == 0:
        flag = 1

f_out.close()